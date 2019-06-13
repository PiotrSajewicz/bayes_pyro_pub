import pyro
import torch
import torch.nn
import numpy as np
import matplotlib
import pyro.optim as optim
import pyro.distributions as dist
from torch.distributions import constraints, Normal
import seaborn as sns
from pyro.infer import SVI, Trace_ELBO

matplotlib.use("TkAgg")
pyro.set_rng_seed(123456)
pyro.enable_validation(True)

# changing lr on my machine often causes NaN's, solution: change seed, prior or lr and pray to god
lr = 0.05
num_iterations = 1000

# load dataset
df = sns.load_dataset('iris')
df.loc[df['species'].values == 'setosa', df.columns == 'species'] = 0
df.loc[df['species'].values == 'virginica', df.columns == 'species'] = 1
df.loc[df['species'].values == 'versicolor', df.columns == 'species'] = 2
df.head()
K = 3
D = 4


def bayes_probs(x, mean, std, pk):
    p_x_c = torch.stack(
        [
            torch.prod
                (
                torch.stack(
                    [
                        torch.exp(Normal(loc=mean[d, k], scale=softplus(std[d, k])).log_prob(x[:, d]))
                        for d in range(D)]
                )
            , dim=0)
        for k in range(K)]
    )
    probs = p_x_c.transpose(0, 1) * pk.squeeze()
    return probs


softplus = torch.nn.Softplus()


def model(x_data, y_data):
    # sample pk before plate(K) bc reasons
    pk_prior = pyro.distributions.Dirichlet(torch.ones(K, ))
    pk_sample = pyro.sample('pk_sample', pk_prior)
    with pyro.plate('k', size=K):
        with pyro.plate('d', size=D):
            mean_prior = pyro.distributions.Normal(loc=torch.ones(D, K) * 2, scale=torch.ones(D, K) * 2)
            std_prior = pyro.distributions.Normal(loc=torch.ones(D, K) * 0.5, scale=torch.ones(D, K))
            mean_sample = pyro.sample('mean_sample', mean_prior)
            std_sample = pyro.sample('std_sample', std_prior)

    # softplus here should not be necessary bc one already is inside 'bayes_probs' func
    # it IS here bc otherwise we get NaN's... i dunno why...
    # EDIT: 'softplus(std_sample)' replaced by 'std_sample'
    # tweaking lr fixed this
    lhat = bayes_probs(x_data, mean_sample, std_sample, pk_sample)
    with pyro.plate('plate', len(y_data)):
        cat = pyro.distributions.Categorical(probs=lhat)
        pyro.sample("obs", cat, obs=y_data)


def guide(x_data, y_data):
    mean_mean = pyro.param("u_u", torch.ones(D, K) * 1.5)
    mean_scale = pyro.param("u_s", torch.ones(D, K), constraint=constraints.positive)
    std_mean = pyro.param("s_u", torch.ones(D, K) * 0.6)
    std_scale = pyro.param("s_s", torch.ones(D, K) * 0.5, constraint=constraints.positive)
    pk_concentration = pyro.param("pk", torch.ones(K, ), constraint=constraints.interval(0.01, 1000))

    pk_prior = pyro.distributions.Dirichlet(pk_concentration)
    pk_sample = pyro.sample('pk_sample', pk_prior)
    with pyro.plate('k', size=K):
        with pyro.plate('d', size=D):
            mean_prior = pyro.distributions.Normal(loc=mean_mean, scale=mean_scale)
            std_prior = pyro.distributions.Normal(loc=std_mean, scale=std_scale)
            mean_sample = pyro.sample('mean_sample', mean_prior)
            std_sample = pyro.sample('std_sample', std_prior)

    return mean_sample, std_sample, pk_sample


pyro.clear_param_store()
optimizer = optim.Adam({"lr": lr})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

x = torch.Tensor(df.loc[:, df.columns != 'species'].values)
y = torch.Tensor(df.loc[:, df.columns == 'species'].values)
y = y.reshape((-1, ))

for j in range(num_iterations):
    loss = svi.step(x, y)
    print("Epoch {} Loss {:10.5f},\n {}".format(j, loss, list(pyro.get_param_store().items())))


print('pred:')
params = guide(None, None)
print(params[0])
print(softplus(params[1]))
print(params[2])
prob_y = bayes_probs(x, params[0], params[1], params[2])
pred_y = torch.argmax(prob_y, dim=1)
print(pred_y)
acc = 1.0 - np.count_nonzero((y-pred_y.type(torch.float)).detach().numpy()) / len(pred_y)
print(acc)
