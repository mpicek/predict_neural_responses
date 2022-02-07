#%%
import torch
import torch.nn as nn
import dnn_blocks as bl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def gauss(x, y, mux, muy, A, sigma):
    """compute gaussian function"""
    x = x - mux
    y = y - muy
    return A * torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))


class dog_layer(nn.Module):
    def __init__(self, num_units, input_size_x, input_size_y, with_bias=False):
        """Difference of gaussian (dog) layer
        based on NDN3 Peter Houska implementation: https://github.com/NeuroTheoryUMD/NDN3/blob/master/layer.py
        and HSM model: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004927
        Args:
            input_size_x (int): number of pixels in the x axis
            input_size_y (int): number of pixels in y axis
            num_units (int): number of difference of gaussians units
        """
        super().__init__()
        self.num_units = num_units
        self.input_size_x = input_size_x
        self.input_size_y = input_size_y
        x_arr = torch.linspace(-1, 1, self.input_size_x)
        y_arr = torch.linspace(-1, 1, self.input_size_y)
        X_mesh, Y_mesh = torch.meshgrid(x_arr, y_arr)
        X_mesh = X_mesh.unsqueeze(0)
        Y_mesh = Y_mesh.unsqueeze(0)
        self.bounds = {
            "A": (torch.finfo(float).eps, None),
            "sigma": (
                1 / max(int(0.82 * self.input_size_x), int(0.82 * self.input_size_y)),
                1,
            ),
            "mu": (-0.8, 0.8),
        }
        self.register_buffer("X", X_mesh, persistent=False)
        self.register_buffer("Y", Y_mesh, persistent=False)
        # initialize
        self.mu_x = nn.Parameter(
            torch.zeros(self.num_units, 1, 1).uniform_(*self.bounds["mu"])
        )
        self.mu_y = nn.Parameter(
            torch.zeros(self.num_units, 1, 1).uniform_(*self.bounds["mu"])
        )
        self.A_c = nn.Parameter(
            torch.ones(self.num_units, 1, 1).uniform_(self.bounds["A"][0], 4)
        )
        self.A_s = nn.Parameter(
            torch.ones(self.num_units, 1, 1).uniform_(self.bounds["A"][0], 4)
        )
        self.sigma_1 = nn.Parameter(
            torch.ones(self.num_units, 1, 1).uniform_(*self.bounds["sigma"])
        )
        self.sigma_2 = nn.Parameter(
            torch.ones(self.num_units, 1, 1).uniform_(*self.bounds["sigma"])
        )
        self.with_bias = with_bias
        if self.with_bias == True:
            self.bias = torch.abs(torch.randn([1, num_units]))
            self.bias = nn.Parameter(self.bias)

    def compute_dog(self):
        """clamp parameters to allowed values and generate dog filter

        Returns:
            torch.Tensor : dog filter size=[n_units, input_size_x, input_size_y]
        """
        self.clamp_parameters()
        sigma_c = self.sigma_1
        sigma_s = self.sigma_1 + self.sigma_2
        g_c = gauss(self.X, self.Y, self.mu_x, self.mu_y, self.A_c, sigma_c)
        g_s = gauss(self.X, self.Y, self.mu_x, self.mu_y, self.A_s, sigma_s)
        dog = g_c - g_s
        return dog

    def clamp_parameters(self):
        """clamp parameters to allowed bound values"""
        self.mu_x.data = torch.clamp(self.mu_x, *self.bounds["mu"])
        self.mu_y.data = torch.clamp(self.mu_y, *self.bounds["mu"])
        self.A_c.data = torch.clamp(self.A_c, *self.bounds["A"])
        self.A_s.data = torch.clamp(self.A_s, *self.bounds["A"])
        self.sigma_1.data = torch.clamp(self.sigma_1, *self.bounds["sigma"])
        self.sigma_2.data = torch.clamp(self.sigma_2, *self.bounds["sigma"])

    def get_dog_parameters(self, i=0):
        """get dictionary of dog parameters for given unit

        Args:
            i (int, optional): unit. Defaults to 0.

        Returns:
            dict: dictionary containing parameters
        """
        params = {
            "A_c": self.A_c[i].squeeze().detach().cpu().numpy(),
            "A_s": self.A_s[i].squeeze().detach().cpu().numpy(),
            "sigma_c": self.sigma_1[i].squeeze().detach().cpu().numpy(),
            "sigma_s": (self.sigma_1[i] + self.sigma_2[i])
            .squeeze()
            .detach()
            .cpu()
            .numpy(),
        }
        return params

    def plot_dog(self, n=0):
        """plot the nth difference of gaussians unit filter

        Args:
            n (int, optional): number of the unit filters to plot. Defaults to 0.
        """

        dog = self.compute_dog()[n].detach().cpu().numpy()
        max_abs_dog = np.max(np.abs(dog))
        p = self.get_dog_parameters(n)
        plt.imshow(dog, cmap=cm.coolwarm)
        plt.title(
            f"A_c = {p['A_c']:.2f},\n"
            + f"A_s = {p['A_s']:.2f},\n"
            + f"sigma_c = {p['sigma_c']:.2f},\n"
            + f"sigma_s = {p['sigma_s']:.2f},\n"
            + f"max = {np.max(dog):.2f},\n"
            + f"min = {np.min(dog):.2f}",
            loc="left",
        )
        plt.clim(-max_abs_dog, max_abs_dog)
        plt.colorbar()
        plt.show()

    def plot_all_dogs(self):
        """plot first n dog unit filters

        Args:
            n (int, optional): number of filters to plot. Defaults to 1.
        """

        n = self.num_units
        size_ax = int(np.ceil(np.sqrt(n)))
        fig, ax = plt.subplots(size_ax, size_ax, figsize=(size_ax * 5, size_ax * 5))
        imgs = []
        for i in range(n):
            dog = self.compute_dog()[i].detach().cpu().numpy()
            max_abs_dog = np.max(np.abs(dog))
            p = self.get_dog_parameters(i)

            pos_x = int(i / size_ax)
            pos_y = int(i % size_ax)

            im = ax[pos_x, pos_y].imshow(
                dog, cmap=cm.coolwarm, vmin=-max_abs_dog, vmax=max_abs_dog
            )
            pars = str(
                f"A_c = {p['A_c']:.2f},\n"
                + f"A_s = {p['A_s']:.2f},\n"
                + f"sigma_c = {p['sigma_c']:.2f},\n"
                + f"sigma_s = {p['sigma_s']:.2f},\n"
                + f"max = {np.max(dog):.2f},\n"
                + f"min = {np.min(dog):.2f}"
            )
            ax[pos_x, pos_y].set_title(pars, size="x-small")
            fig.colorbar(im, ax=ax[pos_x, pos_y])
        plt.show()

    def forward(self, x):
        dogs = self.compute_dog()

        x = torch.tensordot(x.squeeze(), dogs, dims=((1, 2), (1, 2)))

        if self.with_bias == True:
            x = x + self.bias
        return x

