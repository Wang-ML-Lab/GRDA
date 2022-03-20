import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from model.modules import (
    FeatureNet,
    PredNet,
    GNet,
    ClassDiscNet,
    CondClassDiscNet,
    DiscNet,
    GraphDNet,
)
import pickle
from visdom import Visdom

# ===========================================================================================================


def to_np(x):
    return x.detach().cpu().numpy()


def to_tensor(x, device="cuda"):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
    else:
        x = x.to(device)
    return x


def flat(x):
    n, m = x.shape[:2]
    return x.reshape(n * m, *x.shape[2:])


def write_pickle(data, name):
    with open(name, "wb") as f:
        pickle.dump(data, f)


# ======================================================================================================================


# the base model
class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        # set output format
        np.set_printoptions(suppress=True, precision=6)

        self.opt = opt
        self.device = opt.device
        self.batch_size = opt.batch_size
        # visualizaiton
        self.use_visdom = opt.use_visdom
        self.use_g_encode = opt.use_g_encode
        if opt.use_visdom:
            self.env = Visdom(port=opt.visdom_port)
            self.test_pane = dict()

        self.num_domain = opt.num_domain
        if self.opt.test_on_all_dmn:
            self.test_dmn_num = self.num_domain
        else:
            self.test_dmn_num = self.opt.tgt_dmn_num

        self.train_log = self.opt.outf + "/loss.log"
        self.model_path = opt.outf + "/model.pth"
        self.out_pic_f = opt.outf + "/plt_pic"
        if not os.path.exists(self.opt.outf):
            os.mkdir(self.opt.outf)
        if not os.path.exists(self.out_pic_f):
            os.mkdir(self.out_pic_f)
        with open(self.train_log, "w") as f:
            f.write("log start!\n")

        mask_list = np.zeros(opt.num_domain)
        mask_list[opt.src_domain] = 1
        self.domain_mask = torch.IntTensor(mask_list).to(
            opt.device
        )

    def learn(self, epoch, dataloader):
        self.train()
        if self.use_g_encode:
            self.netG.eval()
        self.epoch = epoch
        loss_values = {loss: 0 for loss in self.loss_names}

        count = 0
        for data in dataloader:
            count += 1
            self.__set_input__(data)
            self.__train_forward__()
            new_loss_values = self.__optimize__()

            # for the loss visualization
            for key, loss in new_loss_values.items():
                loss_values[key] += loss

        for key, _ in new_loss_values.items():
            loss_values[key] /= count

        if self.use_visdom:
            self.__vis_loss__(loss_values)

        if (self.epoch + 1) % 10 == 0:
            print("epoch {}: {}".format(self.epoch, loss_values))

        # learning rate decay
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()

    def test(self, epoch, dataloader):
        self.eval()

        acc_curve = []
        l_x = []
        l_domain = []
        l_label = []
        l_encode = []
        z_seq = 0

        for data in dataloader:
            self.__set_input__(data)

            # forward
            with torch.no_grad():
                self.__test_forward__()

            acc_curve.append(
                self.g_seq.eq(self.y_seq)
                .to(torch.float)
                .mean(-1, keepdim=True)
            )
            l_x.append(to_np(self.x_seq))
            l_domain.append(to_np(self.domain_seq))
            l_encode.append(to_np(self.e_seq))
            l_label.append(to_np(self.g_seq))

        x_all = np.concatenate(l_x, axis=1)
        e_all = np.concatenate(l_encode, axis=1)
        domain_all = np.concatenate(l_domain, axis=1)
        label_all = np.concatenate(l_label, axis=1)

        z_seq = to_np(self.z_seq)
        z_seq_all = z_seq[
            0 : self.batch_size * self.test_dmn_num : self.batch_size, :
        ]

        d_all = dict()

        d_all["data"] = flat(x_all)
        d_all["domain"] = flat(domain_all)
        d_all["label"] = flat(label_all)
        d_all["encodeing"] = flat(e_all)
        d_all["z"] = z_seq_all

        acc = to_np(torch.cat(acc_curve, 1).mean(-1))
        test_acc = (
            (acc.sum() - acc[self.opt.src_domain].sum())
            / (self.opt.num_target)
            * 100
        )
        acc_msg = "[Test][{}] Accuracy: total average {:.1f}, test average {:.1f}, in each domain {}".format(
            epoch, acc.mean() * 100, test_acc, np.around(acc * 100, decimals=1)
        )
        self.__log_write__(acc_msg)
        if self.use_visdom:
            self.__vis_test_error__(test_acc, "test acc")

        d_all["acc_msg"] = acc_msg

        write_pickle(d_all, self.opt.outf + "/" + str(epoch) + "_pred.pkl")

    def __vis_test_error__(self, loss, title):
        if self.epoch == self.opt.test_interval - 1:
            # initialize
            self.test_pane[title] = self.env.line(
                X=np.array([self.epoch]),
                Y=np.array([loss]),
                opts=dict(title=title),
            )
        else:
            self.env.line(
                X=np.array([self.epoch]),
                Y=np.array([loss]),
                win=self.test_pane[title],
                update="append",
            )

    def save(self):
        torch.save(self.state_dict(), self.model_path)

    def __set_input__(self, data, train=True):
        """
        :param
            x_seq: Number of domain x Batch size x  Data dim
            y_seq: Number of domain x Batch size x Predict Data dim
            one_hot_seq: Number of domain x Batch size x Number of vertices (domains)
            domain_seq: Number of domain x Batch size x domain dim (1)
        """
        # the domain seq is in d3!!
        x_seq, y_seq, domain_seq = (
            [d[0][None, :, :] for d in data],
            [d[1][None, :] for d in data],
            [d[2][None, :] for d in data],
        )
        self.x_seq = torch.cat(x_seq, 0).to(self.device)
        self.y_seq = torch.cat(y_seq, 0).to(self.device)
        self.domain_seq = torch.cat(domain_seq, 0).to(self.device)
        self.tmp_batch_size = self.x_seq.shape[1]
        one_hot_seq = [
            torch.nn.functional.one_hot(d[2], self.num_domain)
            for d in data
        ]

        if train:
            self.one_hot_seq = (
                torch.cat(one_hot_seq, 0)
                .reshape(self.num_domain, self.tmp_batch_size, -1)
                .to(self.device)
            )
        else:
            self.one_hot_seq = (
                torch.cat(one_hot_seq, 0)
                .reshape(self.test_dmn_num, self.tmp_batch_size, -1)
                .to(self.device)
            )

    def __train_forward__(self):
        pass

    def __test_forward__(self):
        self.z_seq = self.netG(self.one_hot_seq)
        self.e_seq = self.netE(self.x_seq, self.z_seq)  # encoder of the data
        self.f_seq = self.netF(self.e_seq)
        self.g_seq = torch.argmax(
            self.f_seq.detach(), dim=2
        )  # class of the prediction

    def __optimize__(self):
        loss_value = dict()
        if not self.use_g_encode:
            self.loss_G = self.__loss_G__()
        
        if self.opt.lambda_gan != 0:
            self.loss_D = self.__loss_D__()

        self.loss_E, self.loss_E_pred, self.loss_E_gan = self.__loss_EF__()

        if not self.use_g_encode:
            self.optimizer_G.zero_grad()
            self.loss_G.backward(retain_graph=True)
        self.optimizer_D.zero_grad()
        self.loss_D.backward(retain_graph=True)
        self.optimizer_EF.zero_grad()
        self.loss_E.backward()

        if not self.use_g_encode:
            self.optimizer_G.step()
        self.optimizer_D.step()
        self.optimizer_EF.step()

        if not self.use_g_encode:
            loss_value["G"] = self.loss_G.item()
        loss_value["D"], loss_value["E_pred"], loss_value["E_gan"] = \
            self.loss_D.item(), self.loss_E_pred.item(), self.loss_E_gan.item()
        return loss_value

    def __loss_G__(self):
        criterion = nn.BCEWithLogitsLoss()

        sub_graph = self.__sub_graph__(my_sample_v=self.opt.sample_v_g)
        errorG = torch.zeros((1,)).to(self.device)
        sample_v = self.opt.sample_v_g

        for i in range(sample_v):
            v_i = sub_graph[i]
            for j in range(i + 1, sample_v):
                v_j = sub_graph[j]
                label = torch.tensor(self.opt.A[v_i][v_j]).to(self.device)
                output = (
                    self.z_seq[v_i * self.tmp_batch_size]
                    * self.z_seq[v_j * self.tmp_batch_size]
                ).sum()
                errorG += criterion(output, label)

        errorG /= sample_v * (sample_v - 1) / 2
        return errorG

    def __loss_D__(self):
        pass

    def __loss_EF__(self):
        pass

    def __log_write__(self, loss_msg):
        print(loss_msg)
        with open(self.train_log, "a") as f:
            f.write(loss_msg + "\n")

    def __vis_loss__(self, loss_values):
        if self.epoch == 0:
            self.panes = {
                loss_name: self.env.line(
                    X=np.array([self.epoch]),
                    Y=np.array([loss_values[loss_name]]),
                    opts=dict(title="loss for {} on epochs".format(loss_name)),
                )
                for loss_name in self.loss_names
            }
        else:
            for loss_name in self.loss_names:
                self.env.line(
                    X=np.array([self.epoch]),
                    Y=np.array([loss_values[loss_name]]),
                    win=self.panes[loss_name],
                    update="append",
                )

    def __init_weight__(self, net=None):
        if net is None:
            net = self
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                #                 nn.init.normal_(m.weight, mean=0, std=0.1)
                #                 nn.init.xavier_normal_(m.weight, gain=10)
                nn.init.constant_(m.bias, val=0)

    # for graph random sampling:
    def __rand_walk__(self, vis, left_nodes):
        chain_node = []
        node_num = 0
        # choose node
        node_index = np.where(vis == 0)[0]
        st = np.random.choice(node_index)
        vis[st] = 1
        chain_node.append(st)
        left_nodes -= 1
        node_num += 1

        cur_node = st
        while left_nodes > 0:
            nx_node = -1

            node_to_choose = np.where(vis == 0)[0]
            num = node_to_choose.shape[0]
            node_to_choose = np.random.choice(
                node_to_choose, num, replace=False
            )

            for i in node_to_choose:
                if cur_node != i:
                    # have an edge and doesn't visit
                    if self.opt.A[cur_node][i] and not vis[i]:
                        nx_node = i
                        vis[nx_node] = 1
                        chain_node.append(nx_node)
                        left_nodes -= 1
                        node_num += 1
                        break
            if nx_node >= 0:
                cur_node = nx_node
            else:
                break
        return chain_node, node_num

    def __sub_graph__(self, my_sample_v):
        if np.random.randint(0, 2) == 0:
            return np.random.choice(
                self.num_domain, size=my_sample_v, replace=False
            )

        # subsample a chain (or multiple chains in graph)
        left_nodes = my_sample_v
        choosen_node = []
        vis = np.zeros(self.num_domain)
        while left_nodes > 0:
            chain_node, node_num = self.__rand_walk__(vis, left_nodes)
            choosen_node.extend(chain_node)
            left_nodes -= node_num

        return choosen_node


class DANN(BaseModel):
    """
    DANN Model
    """

    def __init__(self, opt):
        super(DANN, self).__init__(opt)
        self.netE = FeatureNet(opt).to(opt.device)
        self.netF = PredNet(opt).to(opt.device)
        self.netG = GNet(opt).to(opt.device)
        self.netD = ClassDiscNet(opt).to(opt.device)

        self.__init_weight__()
        EF_parameters = list(self.netE.parameters()) + list(
            self.netF.parameters()
        )
        self.optimizer_EF = optim.Adam(
            EF_parameters, lr=opt.lr_e, betas=(opt.beta1, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999)
        )
        if not self.use_g_encode:
            self.optimizer_G = optim.Adam(
                self.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999)
            )

        self.lr_scheduler_EF = lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_EF, gamma=0.5 ** (1 / 100)
        )
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_D, gamma=0.5 ** (1 / 100)
        )
        if not self.use_g_encode:
            self.lr_scheduler_G = lr_scheduler.ExponentialLR(
                optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100)
            )
            self.lr_schedulers = [
                self.lr_scheduler_EF,
                self.lr_scheduler_D,
                self.lr_scheduler_G,
            ]
        else:
            self.lr_schedulers = [self.lr_scheduler_EF, self.lr_scheduler_D]
        self.loss_names = ["E_pred", "E_gan", "D", "G"]
        if self.use_g_encode:
            self.loss_names.remove("G")

        self.lambda_gan = self.opt.lambda_gan

    def __train_forward__(self):
        self.z_seq = self.netG(self.one_hot_seq)
        self.e_seq = self.netE(self.x_seq, self.z_seq)  # encoder of the data
        self.f_seq = self.netF(self.e_seq)
        self.d_seq = self.netD(self.e_seq)

    def __loss_D__(self):
        return F.nll_loss(flat(self.d_seq), flat(self.domain_seq))

    def __loss_EF__(self):
        # we have already calculate D loss before EF loss. Directly use it.
        loss_E_gan = - self.loss_D

        y_seq_source = self.y_seq[self.domain_mask == 1]
        f_seq_source = self.f_seq[self.domain_mask == 1]

        loss_E_pred = F.nll_loss(
            flat(f_seq_source), flat(y_seq_source)
        )

        loss_E = loss_E_gan * self.lambda_gan + loss_E_pred

        return loss_E, loss_E_pred, loss_E_gan


class CDANN(BaseModel):
    """
    CDANN Model
    """

    def __init__(self, opt):
        super(CDANN, self).__init__(opt)
        self.netE = FeatureNet(opt).to(opt.device)
        self.netF = PredNet(opt).to(opt.device)
        self.netG = GNet(opt).to(opt.device)
        self.netD = CondClassDiscNet(opt).to(opt.device)

        self.__init_weight__()
        EF_parameters = list(self.netE.parameters()) + list(
            self.netF.parameters()
        )
        self.optimizer_EF = optim.Adam(
            EF_parameters, lr=opt.lr_e, betas=(opt.beta1, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999)
        )

        if not self.use_g_encode:
            self.optimizer_G = optim.Adam(
                self.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999)
            )

        self.lr_scheduler_EF = lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_EF, gamma=0.5 ** (1 / 100)
        )
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_D, gamma=0.5 ** (1 / 100)
        )
        if not self.use_g_encode:
            self.lr_scheduler_G = lr_scheduler.ExponentialLR(
                optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100)
            )
            self.lr_schedulers = [
                self.lr_scheduler_EF,
                self.lr_scheduler_D,
                self.lr_scheduler_G,
            ]
        else:
            self.lr_schedulers = [self.lr_scheduler_EF, self.lr_scheduler_D]
        self.loss_names = ["E_pred", "E_gan", "D", "G"]
        if self.use_g_encode:
            self.loss_names.remove("G")

        self.lambda_gan = self.opt.lambda_gan

    def __train_forward__(self):
        self.z_seq = self.netG(self.one_hot_seq)
        self.e_seq = self.netE(self.x_seq, self.z_seq)  # encoder of the data
        self.f_seq = self.netF(self.e_seq)
        self.f_seq_sig = torch.sigmoid(self.f_seq.detach())

    def __loss_D__(self):
        d_seq = self.netD(self.e_seq.detach(), self.f_seq_sig.detach())
        return F.nll_loss(flat(d_seq), flat(self.domain_seq))

    def __loss_EF__(self):
        d_seq = self.netD(self.e_seq, self.f_seq_sig.detach())

        loss_E_gan = -F.nll_loss(flat(d_seq), flat(self.domain_seq))

        y_seq_source = self.y_seq[self.domain_mask == 1]
        f_seq_source = self.f_seq[self.domain_mask == 1]

        loss_E_pred = F.nll_loss(
            flat(f_seq_source), flat(y_seq_source)
        )

        loss_E = loss_E_gan * self.lambda_gan + loss_E_pred
        return loss_E, loss_E_pred, loss_E_gan


class ADDA(BaseModel):
    def __init__(self, opt):
        super(ADDA, self).__init__(opt)
        self.netE = FeatureNet(opt).to(opt.device)
        self.netF = PredNet(opt).to(opt.device)
        self.netG = GNet(opt).to(opt.device)
        self.netD = DiscNet(opt).to(opt.device)
        self.__init_weight__()
        EF_parameters = list(self.netE.parameters()) + list(
            self.netF.parameters()
        )
        self.optimizer_EF = optim.Adam(
            EF_parameters, lr=opt.lr_e, betas=(opt.beta1, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999)
        )

        if not self.use_g_encode:
            self.optimizer_G = optim.Adam(
                self.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999)
            )

        self.lr_scheduler_EF = lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_EF, gamma=0.5 ** (1 / 100)
        )
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_D, gamma=0.5 ** (1 / 100)
        )
        if not self.use_g_encode:
            self.lr_scheduler_G = lr_scheduler.ExponentialLR(
                optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100)
            )
            self.lr_schedulers = [
                self.lr_scheduler_EF,
                self.lr_scheduler_D,
                self.lr_scheduler_G,
            ]
        else:
            self.lr_schedulers = [self.lr_scheduler_EF, self.lr_scheduler_D]
        self.loss_names = ["E_pred", "E_gan", "D", "G"]
        if self.use_g_encode:
            self.loss_names.remove("G")

    def __train_forward__(self):
        self.z_seq = self.netG(self.one_hot_seq)
        self.e_seq = self.netE(self.x_seq, self.z_seq)  # encoder of the data
        self.f_seq = self.netF(self.e_seq)

    def __loss_D__(self):
        d_seq = self.netD(self.e_seq.detach())
        d_seq_source = d_seq[self.domain_mask == 1]
        d_seq_target = d_seq[self.domain_mask == 0]
        # D: discriminator loss from classifying source v.s. target
        loss_D = (
            -torch.log(d_seq_source + 1e-10).mean()
            - torch.log(1 - d_seq_target + 1e-10).mean()
        )
        return loss_D

    def __loss_EF__(self):
        d_seq = self.netD(self.e_seq)
        d_seq_target = d_seq[self.domain_mask == 0]
        loss_E_gan = -torch.log(d_seq_target + 1e-10).mean()
        # E_pred: encoder loss from prediction the label
        y_seq_source = self.y_seq[self.domain_mask == 1]
        f_seq_source = self.f_seq[self.domain_mask == 1]
        loss_E_pred = F.nll_loss(
            flat(f_seq_source), flat(y_seq_source)
        )

        loss_E = loss_E_gan * self.opt.lambda_gan + loss_E_pred

        return loss_E, loss_E_pred, loss_E_gan


class MDD(BaseModel):
    """
    Margin Disparity Discrepancy
    """

    def __init__(self, opt):
        super(MDD, self).__init__(opt)
        self.netE = FeatureNet(opt).to(opt.device)
        self.netF = PredNet(opt).to(opt.device)
        self.netG = GNet(opt).to(opt.device)
        self.netD = PredNet(opt).to(opt.device)
        self.__init_weight__()
        EF_parameters = list(self.netE.parameters()) + list(
            self.netF.parameters()
        )
        self.optimizer_EF = optim.Adam(
            EF_parameters, lr=opt.lr_e, betas=(opt.beta1, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999)
        )
        if not self.use_g_encode:
            self.optimizer_G = optim.Adam(
                self.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999)
            )

        self.lr_scheduler_EF = lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_EF, gamma=0.5 ** (1 / 100)
        )
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_D, gamma=0.5 ** (1 / 100)
        )
        if not self.use_g_encode:
            self.lr_scheduler_G = lr_scheduler.ExponentialLR(
                optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100)
            )
            self.lr_schedulers = [
                self.lr_scheduler_EF,
                self.lr_scheduler_D,
                self.lr_scheduler_G,
            ]
        else:
            self.lr_schedulers = [self.lr_scheduler_EF, self.lr_scheduler_D]
        self.loss_names = ["E_pred", "E_adv", "ADV_src", "ADV_tgt", "G"]
        if self.use_g_encode:
            self.loss_names.remove("G")

        self.lambda_src = opt.lambda_src
        self.lambda_tgt = opt.lambda_tgt

    def __train_forward__(self):
        self.z_seq = self.netG(self.one_hot_seq)
        self.e_seq = self.netE(self.x_seq, self.z_seq)  # encoder of the data
        self.f_seq = self.netF(self.e_seq)
        self.g_seq = torch.argmax(
            self.f_seq.detach(), dim=2
        )  # class of the prediction

    def __optimize__(self):
        loss_value = dict()
        if not self.use_g_encode:
            self.loss_G = self.__loss_G__()
        
        self.loss_D, self.loss_ADV_src, self.loss_ADV_tgt = self.__loss_D__()
        self.loss_E, self.loss_E_pred, self.loss_E_adv = self.__loss_EF__()

        if not self.use_g_encode:
            self.optimizer_G.zero_grad()
        self.loss_G.backward(retain_graph=True)
        self.optimizer_D.zero_grad()
        self.loss_D.backward(retain_graph=True)
        self.optimizer_EF.zero_grad()
        self.loss_E.backward()

        if not self.use_g_encode:
            self.optimizer_G.step()
        self.optimizer_D.step()
        self.optimizer_EF.step()

        if not self.use_g_encode:
            loss_value["G"] = self.loss_G.item()
        loss_value["ADV_src"], loss_value["ADV_tgt"], loss_value["E_pred"], loss_value["E_adv"] = \
            self.loss_ADV_src.item(), self.loss_ADV_tgt.item(), self.loss_E_pred.item(), self.loss_E_adv.item()
        return loss_value


        # loss_value = dict()
        # if not self.use_g_encode:
        #     loss_value["G"] = self.__optimize_G__()
        # # if self.opt.lambda_gan != 0:
        # #     (
        # #         loss_value["ADV_src"],
        # #         loss_value["ADV_tgt"],
        # #     ) = self.__optimize_D__()
        # # else:
        # #     loss_value["ADV_src"], loss_value["ADV_tgt"] = 0
        # # (
        #     loss_value["E_pred"],
        #     loss_value["E_adv"],
        # ) = self.__optimize_EF__()
        # return loss_value

    def __loss_D__(self):
        # self.optimizer_D.zero_grad()

        # # backward process:
        # self.loss_D.backward(retain_graph=True)
        # return self.loss_D.item()
        f_adv, f_adv_softmax = self.netD(
            self.e_seq.detach(), return_softmax=True
        )
        # agreement with netF on source domain
        loss_ADV_src = F.nll_loss(
            flat(f_adv[self.domain_mask == 1]),
            flat(self.g_seq[self.domain_mask == 1]),
        )
        f_adv_tgt = torch.log(
            1 - f_adv_softmax[self.domain_mask == 0] + 1e-10
        )
        # disagreement with netF on target domain
        loss_ADV_tgt = F.nll_loss(
            flat(f_adv_tgt), flat(self.g_seq[self.domain_mask == 0])
        )
        # minimize the agreement on source domain while maximize the disagreement on target domain
        loss_D = (
            loss_ADV_src * self.lambda_src
            + loss_ADV_tgt * self.lambda_tgt
        ) / (self.lambda_src + self.lambda_tgt)

        return loss_D, loss_ADV_src, loss_ADV_tgt

    def __loss_EF__(self):
        loss_E_pred = F.nll_loss(
            flat(self.f_seq[self.domain_mask == 1]),
            flat(self.y_seq[self.domain_mask == 1]),
        )

        f_adv, f_adv_softmax = self.netD(
            self.e_seq, return_softmax=True
        )
        loss_ADV_src = F.nll_loss(
            flat(f_adv[self.domain_mask == 1]),
            flat(self.g_seq[self.domain_mask == 1]),
        )
        f_adv_tgt = torch.log(
            1 - f_adv_softmax[self.domain_mask == 0] + 1e-10
        )
        loss_ADV_tgt = F.nll_loss(
            flat(f_adv_tgt), flat(self.g_seq[self.domain_mask == 0])
        )
        loss_E_adv = -(
            loss_ADV_src * self.lambda_src
            + loss_ADV_tgt * self.lambda_tgt
        ) / (self.lambda_src + self.lambda_tgt)
        loss_E = loss_E_pred + self.opt.lambda_gan * loss_E_adv

        return loss_E, loss_E_pred, loss_E_adv


class GDA(BaseModel):
    """
    GDA Model
    """

    def __init__(self, opt):
        super(GDA, self).__init__(opt)
        self.netE = FeatureNet(opt).to(opt.device)
        self.netF = PredNet(opt).to(opt.device)
        self.netG = GNet(opt).to(opt.device)
        self.netD = GraphDNet(opt).to(opt.device)
        self.__init_weight__()

        EF_parameters = list(self.netE.parameters()) + list(
            self.netF.parameters()
        )
        self.optimizer_EF = optim.Adam(
            EF_parameters, lr=opt.lr_e, betas=(opt.beta1, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999)
        )
        if not self.use_g_encode:
            self.optimizer_G = optim.Adam(
                self.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999)
            )

        self.lr_scheduler_EF = lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_EF, gamma=0.5 ** (1 / 100)
        )
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_D, gamma=0.5 ** (1 / 100)
        )
        if not self.use_g_encode:
            self.lr_scheduler_G = lr_scheduler.ExponentialLR(
                optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100)
            )
            self.lr_schedulers = [
                self.lr_scheduler_EF,
                self.lr_scheduler_D,
                self.lr_scheduler_G,
            ]
        else:
            self.lr_schedulers = [self.lr_scheduler_EF, self.lr_scheduler_D]
        self.loss_names = ["E_pred", "E_gan", "D", "G"]
        if self.use_g_encode:
            self.loss_names.remove("G")

    def __train_forward__(self):
        self.z_seq = self.netG(self.one_hot_seq)
        self.e_seq = self.netE(self.x_seq, self.z_seq)  # encoder of the data
        self.f_seq = self.netF(self.e_seq)  # prediction
        self.d_seq = self.netD(self.e_seq)

    def __loss_EF__(self):
        # we have already got loss D
        if self.opt.lambda_gan != 0:
            loss_E_gan = -self.loss_D
        else:
            loss_E_gan = torch.tensor(
                0, dtype=torch.float, device=self.opt.device
            )

        y_seq_source = self.y_seq[self.domain_mask == 1]
        f_seq_source = self.f_seq[self.domain_mask == 1]

        loss_E_pred = F.nll_loss(flat(f_seq_source), flat(y_seq_source))

        loss_E = loss_E_gan * self.opt.lambda_gan + loss_E_pred

        return loss_E, loss_E_pred, loss_E_gan

    def __loss_D__(self):
        criterion = nn.BCEWithLogitsLoss()
        d = self.d_seq
        # random pick subchain and optimize the D
        # balance coefficient is calculate by pos/neg ratio
        sub_graph = self.__sub_graph__(my_sample_v=self.opt.sample_v)

        errorD_connected = torch.zeros((1,)).to(self.device)  # .double()
        errorD_disconnected = torch.zeros((1,)).to(self.device)  # .double()

        count_connected = 0
        count_disconnected = 0

        for i in range(self.opt.sample_v):
            v_i = sub_graph[i]
            # no self loop version!!
            for j in range(i + 1, self.opt.sample_v):
                v_j = sub_graph[j]
                label = torch.full(
                    (self.tmp_batch_size,),
                    self.opt.A[v_i][v_j],
                    device=self.device,
                )
                # dot product
                if v_i == v_j:
                    idx = torch.randperm(self.tmp_batch_size)
                    output = (d[v_i][idx] * d[v_j]).sum(1)
                else:
                    output = (d[v_i] * d[v_j]).sum(1)

                if self.opt.A[v_i][v_j]:  # connected
                    errorD_connected += criterion(output, label)
                    count_connected += 1
                else:
                    errorD_disconnected += criterion(output, label)
                    count_disconnected += 1

        errorD = 0.5 * (
            errorD_connected / count_connected
            + errorD_disconnected / count_disconnected
        )
        # this is a loss balance
        return errorD * self.num_domain
