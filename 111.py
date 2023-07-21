import torch
import torch.nn as nn
from torch.optim import Adagrad, RMSprop, Adadelta, Adam, AdamW, NAdam


class Base:
    def __init__(self, params, lr=0.01, eps=1e-10, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay

        self.steps = 1

    @torch.no_grad()
    def step(self):
        raise NotImplementedError


class MyAdagrad(Base):
    def __init__(self, params, lr=0.01, eps=1e-10, weight_decay=0.0, initial_accumulator_value=0.0, lr_decay=0.0):
        super(MyAdagrad, self).__init__(params, lr, eps, weight_decay)
        self.initial_accumulator_value = initial_accumulator_value
        self.lr_decay = lr_decay

        self.state_sum_t = {k: torch.zeros_like(v) for k, v in self.params.items()}

    @torch.no_grad()
    def step(self):
        for k, weight in self.params.items():
            grad_ = weight.grad
            lr_ = self.lr / (1 + (self.steps - 1) * self.lr_decay)
            if self.weight_decay != 0:
                grad_ += self.weight_decay * weight
            
            state_sum_t = self.state_sum_t[k] + torch.pow(grad_, 2)
            weight -= lr_ * grad_ / (torch.sqrt(state_sum_t) + self.eps)

            self.state_sum_t[k] = state_sum_t

        self.steps += 1


class MyRMSprop(Base):
    def __init__(self, params, lr=0.01, eps=1e-8, weight_decay=0.0, momentum=0.0, alpha=0.99, centered=False):
        super(MyRMSprop, self).__init__(params, lr, eps, weight_decay)
        self.momentum = momentum
        self.alpha = alpha
        self.centered = centered

        self.v_t = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.b_t = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.grad_avg = {k: torch.zeros_like(v) for k, v in self.params.items()}

    @torch.no_grad()
    def step(self):
        for k, weight in self.params.items():
            grad_ = weight.grad
            if self.weight_decay != 0:
                grad_ += self.weight_decay * weight

            self.v_t[k] = self.alpha * self.v_t[k] + (1 - self.alpha) * torch.pow(grad_, 2)
            v_t_heat = self.v_t[k]

            if self.centered:
                self.grad_avg[k] = self.alpha * self.grad_avg[k] + (1 - self.alpha) * grad_
                v_t_heat -= torch.pow(self.grad_avg[k], 2)

            if self.momentum > 0:
                self.b_t[k] = self.momentum * self.b_t[k] + grad_ / (torch.sqrt(v_t_heat) + self.eps)
                weight -= self.lr * self.b_t[k]
            else:
                weight -= self.lr * grad_ / (torch.sqrt(v_t_heat) + self.eps)

        self.steps += 1


class MyAdadelta(Base):
    def __init__(self, params, lr=1.0, eps=1e-6, weight_decay=0.0, rho=0.9):
        super(MyAdadelta, self).__init__(params, lr, eps, weight_decay)
        self.rho = rho

        self.v_t = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.mu_t = {k: torch.zeros_like(v) for k, v in self.params.items()}

        self.steps = 1

    @torch.no_grad()
    def step(self):
        for k, weight in self.params.items():
            grad_ = weight.grad
            if self.weight_decay != 0:
                grad_ += self.weight_decay * weight

            self.v_t[k] = self.rho * self.v_t[k] + (1 - self.rho) * torch.pow(grad_, 2)
            delta_x = grad_ * (torch.sqrt(self.mu_t[k] + self.eps) / torch.sqrt(self.v_t[k] + self.eps))
            self.mu_t[k] = self.rho * self.mu_t[k] + (1 - self.rho) * torch.pow(delta_x, 2)

            weight -= self.lr * delta_x

        self.steps += 1


class MyAdam(Base):
    def __init__(self, params, lr=1e-3, eps=1e-8, weight_decay=0, betas=(0.9, 0.999), amsgrad=False, maximize=False):
        super(MyAdam, self).__init__(params, lr, eps, weight_decay)
        self.betas = betas
        self.amsgrad = amsgrad
        self.maximize = maximize

        self.m_t = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.v_t = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.v_t_heat_max = {k: torch.zeros_like(v) for k, v in self.params.items()}

    @torch.no_grad()
    def step(self):
        for k, weight in self.params.items():
            if self.maximize:
                grad_ = -weight.grad
            else:
                grad_ = weight.grad

            if self.weight_decay != 0:
                grad_ += self.weight_decay * weight

            self.m_t[k] = self.betas[0] * self.m_t[k] + (1 - self.betas[0]) * grad_
            self.v_t[k] = self.betas[1] * self.v_t[k] + (1 - self.betas[1]) * torch.pow(grad_, 2)

            m_t_heat = self.m_t[k] / (1 - self.betas[0] ** self.steps)
            v_t_heat = self.v_t[k] / (1 - self.betas[1] ** self.steps)

            if self.amsgrad:
                self.v_t_heat_max[k] = max(self.v_t_heat_max[k], v_t_heat)
                weight -= self.lr * m_t_heat / (torch.sqrt(self.v_t_heat_max[k] + self.eps))
            else:
                weight -= self.lr * m_t_heat / (torch.sqrt(v_t_heat) + self.eps)

        self.steps += 1


class MyAdamW(Base):
    def __init__(self, params, lr=1e-3, eps=1e-8, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, maximize=False):
        super(MyAdamW, self).__init__(params, lr, eps, weight_decay)
        self.betas = betas
        self.amsgrad = amsgrad
        self.maximize = maximize

        self.m_t = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.v_t = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.v_t_heat_max = {k: torch.zeros_like(v) for k, v in self.params.items()}

    @torch.no_grad()
    def step(self):
        for k, weight in self.params.items():
            if self.maximize:
                grad_ = -weight.grad
            else:
                grad_ = weight.grad

            weight -= self.lr * self.weight_decay * weight

            self.m_t[k] = self.betas[0] * self.m_t[k] + (1 - self.betas[0]) * grad_
            self.v_t[k] = self.betas[1] * self.v_t[k] + (1 - self.betas[1]) * torch.pow(grad_, 2)

            m_t_heat = self.m_t[k] / (1 - self.betas[0] ** self.steps)
            v_t_heat = self.v_t[k] / (1 - self.betas[1] ** self.steps)

            if self.amsgrad:
                self.v_t_heat_max[k] = max(self.v_t_heat_max[k], v_t_heat)
                weight -= self.lr * m_t_heat / (torch.sqrt(self.v_t_heat_max[k] + self.eps))
            else:
                weight -= self.lr * m_t_heat / (torch.sqrt(v_t_heat) + self.eps)

        self.steps += 1


class MyNAdam(Base):
    def __init__(self, params, lr=2e-3, eps=1e-8, weight_decay=0.0, betas=(0.9, 0.999), momentum_decay=4e-3):
        super(MyNAdam, self).__init__(params, lr, eps, weight_decay)
        self.betas = betas
        self.momentum_decay = momentum_decay

        self.m_t = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.v_t = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.mu_multi = torch.tensor(1.0)

    @torch.no_grad()
    def step(self):
        mu_t = self.betas[0] * (1 - 0.5 * 0.96 ** (self.steps * self.momentum_decay))
        mu_t_1 = self.betas[0] * (1 - 0.5 * 0.96 ** ((self.steps + 1) * self.momentum_decay))
        self.mu_multi *= mu_t
        mu_multi_1 = self.mu_multi * mu_t_1

        for k, weight in self.params.items():
            grad_ = weight.grad
            if self.weight_decay != 0:
                grad_ += self.weight_decay * weight

            self.m_t[k] = self.betas[0] * self.m_t[k] + (1 - self.betas[0]) * grad_
            self.v_t[k] = self.betas[1] * self.v_t[k] + (1 - self.betas[1]) * torch.pow(grad_, 2)

            m_t_heat = (mu_t_1 * self.m_t[k] / (1 - mu_multi_1)) + ((1 - mu_t) * grad_ / (1 - self.mu_multi))
            v_t_heat = self.v_t[k] / (1 - self.betas[1] ** self.steps)

            weight -= self.lr * m_t_heat / (torch.sqrt(v_t_heat) + self.eps)

        self.steps += 1


class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super(MLP, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features,
                                bias=True)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y):
        outputs = self.linear(x)
        loss = self.loss_fn(outputs, y)
        return outputs, loss


if __name__ == '__main__':

    optimizer_key = 'nadam'
    my_optimizer_dict = {'adagrad': MyAdagrad, 'rmsprop': MyRMSprop, 'adadelta': MyAdadelta,
                         'adam': MyAdam, 'adamw': MyAdamW, 'nadam': MyNAdam}
    torch_optimizer_dict = {'adagrad': Adagrad, 'rmsprop': RMSprop, 'adadelta': Adadelta,
                            'adam': Adam, 'adamw': AdamW, 'nadam': NAdam}

    torch.manual_seed(72)

    model = MLP(in_features=2, out_features=4)
    inputs_ = torch.tensor([[1, 1], [1, 1]], dtype=torch.float)
    label = torch.tensor([0, 1]).long()

    optimizer = my_optimizer_dict[optimizer_key](dict(model.named_parameters()), weight_decay=0.01)
    # optimizer = torch_optimizer_dict[optimizer_key](model.parameters(), weight_decay=0.01)

    for _ in range(10):
        out_, loss = model(inputs_, label)
        loss.backward()
        optimizer.step()
        print(model.linear.bias)
        print('-' * 100)


