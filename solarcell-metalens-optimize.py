'''
( real(exp(1/1 * h * ...)) + real(exp(1/2 * h * ...)) ++ ) ^ 2

'''


import torcwa
import csv
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

'''
Script settings
'''
logging = False
debug = True
verbose = 0
handle_instability = True

'''
Generator settings
'''
num_images = 2
hidden_dimension = 10
noise_dimension = 3

# TODO add intelligent model to simulation
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.noise = torch.rand(size=(num_images, noise_dimension))
        self.FC = nn.Sequential(nn.Linear(in_features=noise_dimension, out_features=hidden_dimension), nn.ReLU(
        ), nn.Linear(in_features=hidden_dimension, out_features=100))

    def forward(self):
        return self.FC(self.noise)


h = 6.626070e-34  # Js Planck's constant
c = 2.997925e8  # m/s speed of light
k_B = 1.380649e-23  # J/K Boltzmann constant
q = 1.602176e-19  # C elementary charge


def Blackbody(lambda_i, T): return (2*h*c**2) / \
    ((np.exp((h*c)/(k_B*T*lambda_i*1e-9))-1)*lambda_i**5)*1e32


def nb_B(lambda_i, T): return (2*c) / \
    ((np.exp((h*c)/(k_B*T*lambda_i*1e-9))-1)*lambda_i**4)*1e23


wavelengths = torch.linspace(350, 3000, 2651)  # Issue when this goes past 99?

T_e = 2073.15  # K emitter temperature
B_i = Blackbody(wavelengths, T_e)  # 2073.15K blackbody
nb_B_e = nb_B(wavelengths, T_e)  # 2073.15K photon
T_PV = 300  # K PV temperature
nb_B_PV = nb_B(wavelengths, T_PV)  # 300K photon

def IQE(wavelength, e_g):
    lambda_g = np.ceil(1240 / e_g)
    if (lambda_g > wavelength[-1]):
        l_index = wavelength[-1]
    else:
        l_index = torch.where(wavelength >= lambda_g)[0][0]
    IQE = torch.ones(len(wavelength))
    for i in range(l_index, len(wavelength)):
        IQE[i] = 0
    return IQE


def JV(em, IQE, lambda_i, T_emitter, iteration, image_index):
    em = em.squeeze()
    # J_L = q * np.sum(em.detach().numpy() * np.array(nb_B(lambda_i, T_emitter)) * IQE.detach().numpy()) * (lambda_i[1] - lambda_i[0])
    J_L = q * torch.sum(em * nb_B_e * IQE) * (lambda_i[1] - lambda_i[0])
    J_0 = q * torch.sum(nb_B_PV*IQE) * (lambda_i[1] - lambda_i[0])
    V_oc = (k_B*T_PV/q)*torch.log(J_L/J_0+1)
    t = torch.linspace(0, 1, 100)
    V = t * V_oc
    J = J_L-J_0*(torch.exp(q*V/(k_B*T_PV))-1)
    P = V*J

    if logging:
        filename = f'power_image_{image_index}.csv'
        with open(filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([torch.max(P).clone().detach().cpu().numpy()])

    return torch.max(P)


def d_optimize(lambda_i, emissivity_dataset, T_emitter, E_g_PV, iteration, image_index):
    emissivity = emissivity_dataset.squeeze()
    # P_emit = simpson_rule(
    #     emissivity*Blackbody(lambda_i, T_emitter), x=lambda_i)
    P_emit = torch.sum(emissivity*Blackbody(lambda_i, T_emitter)
                       ) * (lambda_i[1] - lambda_i[0])
    print(f'This is P_emit: {P_emit}')
    IQE_PV = IQE(lambda_i, E_g_PV)
    JV_PV = JV(emissivity, IQE_PV, lambda_i, T_emitter, iteration, image_index)
    print(f'This is JV_PV: {JV_PV}')
    FOM = JV_PV / P_emit

    if logging:
        filename = f'FOM_image_{image_index}.csv'
        with open(filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([FOM.clone().detach().cpu().numpy()])

    return FOM


n_all = pd.read_excel(
    "/Users/raymondiacobacci/LeiteLab-6_8/n-allHTMats-2.xlsx")
k_all = pd.read_excel("/Users/raymondiacobacci/LeiteLab-6_8/k-allHTMats.xlsx")

nk_AlN = n_all["AlN"] + 1j * k_all["AlN"]
nk_W = n_all["W"] + 1j * k_all["W"]


sys.path.append('..')

torch.backends.cuda.matmul.allow_tf32 = False
sim_dtype = torch.complex64
geo_dtype = torch.float32
device = torch.device('cpu')

inc_ang = 0.*(np.pi/180)    # radian
azi_ang = 0.*(np.pi/180)    # radian

L = [1000., 1000.]            # nm / nm
torcwa.rcwa_geo.dtype = geo_dtype
torcwa.rcwa_geo.device = device
torcwa.rcwa_geo.Lx = L[0]
torcwa.rcwa_geo.Ly = L[1]
torcwa.rcwa_geo.nx = 200
# TODO bug
useless_transverse_points = 5
torcwa.rcwa_geo.ny = useless_transverse_points
torcwa.rcwa_geo.grid()
torcwa.rcwa_geo.edge_sharpness = 10000.
z = torch.linspace(0, 1000, 1, device=device)
learning_rate = 1e-3

x_axis = torcwa.rcwa_geo.x.cpu()
y_axis = torcwa.rcwa_geo.y.cpu()
z_axis = z.cpu()

order_N = 1
order = [order_N, 1]
order = [0,1]

generator = Generator()
optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
lambda_quad = 1e-5
for iteration in range(100):
    optimizer.zero_grad()
    generated_images = generator()
    total_gradients = []
    for image in range(num_images):
        grid_permittivity = generated_images[image].unsqueeze(
            1).repeat(1, useless_transverse_points)
        grid_permittivity = torch.clamp(grid_permittivity, min=0, max=1)
        grid_permittivity.requires_grad_(True)

        if verbose >= 1:
            np_grid = grid_permittivity.cpu().detach().numpy()
            filename = f"permittivity_iteration_{iteration}_image_{image}.npy"
            np.save(filename, np_grid)

        gradient_array = []
        reflected_array = []

        if debug:
            grid_permittivity *= 0
            grid_permittivity += 1

        for index, wavelength in enumerate(wavelengths):
            if debug:
                with open('logger.txt','a+') as f:
                    f.write(f'Wavelength: {wavelength}\n')
            # wavelength = 1500
            # index = 1150
            # grid_permittivity[50:60] = 0
            grid_permittivity_scaled = grid_permittivity * \
                (nk_AlN[index] ** 2 - 1) + 1
            sim = torcwa.rcwa(freq=1 / wavelength, order=order,
                              L=[500., 500.], dtype=sim_dtype, device=device)
            sim.add_input_layer(eps=1)
            sim.set_incident_angle(inc_ang=inc_ang, azi_ang=azi_ang)
            if not debug:
                sim.add_layer(
                    thickness=473, eps=torch.tensor(nk_AlN[index] ** 2))
            else:
                sim.add_layer(thickness=473, eps=grid_permittivity_scaled)
            sim.add_layer(thickness=5000, eps=torch.tensor(nk_W[index] ** 2))
            if verbose >= 1:
                print('-'*5)
            sim.solve_global_smatrix()

            reflected = torch.sqrt(torch.abs(sim.S_parameters(orders=[0, 0], direction='forward', port='reflection', polarization='xx', ref_order=[
                0, 0])) ** 2 + torch.abs(sim.S_parameters(orders=[0, 0], direction='forward', port='reflection', polarization='yx', ref_order=[0, 0])) ** 2)
            print(f'Transmission: {1 - reflected}')
            # sys.exit()
            reflected_array.append(reflected)
            sim.source_planewave(amplitude=[1., 0.], direction='forward')
            [Ex, Ey, Ez], [Hx, Hy, Hz] = sim.field_xz(
                torcwa.rcwa_geo.x, z, L[1] / 2)
            sim.source_planewave(amplitude=[reflected, 0], direction='forward')
            [Ex_adj, Ey_adj, Ez_adj], [Hx_adj, Hy_adj, Hz_adj] = sim.field_xz(
                torcwa.rcwa_geo.x, z, L[1] / 2)
            dj_de = 1/wavelength**2*torch.real(torch.mul(Ex, Ex_adj))
            gradient_array.append(dj_de)
        gradient_array_stacked = torch.mean(
            torch.stack(gradient_array), dim=(-1))
        transmitted_array_stacked = 1 - torch.stack(reflected_array)
        # TODO bug (numerical instability)
        if handle_instability:
            print(torch.argmax(transmitted_array_stacked))
            print(transmitted_array_stacked[torch.argmax(transmitted_array_stacked)])
            transmitted_array_stacked[torch.argmax(transmitted_array_stacked):torch.argmax(transmitted_array_stacked)+1] = transmitted_array_stacked[torch.argmax(transmitted_array_stacked)-1:torch.argmax(transmitted_array_stacked)]
        if debug:
            print(
                f'Computed transmission coefficients: {transmitted_array_stacked.flatten().detach().numpy()}')
            # transmitted_array_stacked = np.array(np.load('/Users/raymondiacobacci/declan_emissivity.npy'))
        transmitted_array_stacked = torch.tensor(
            transmitted_array_stacked, requires_grad=True)
        FOM = 1-d_optimize(wavelengths, transmitted_array_stacked,
                           T_e, .726, iteration, image)
        # Need to write a test for it to make sure that these calculations (transposed) are correct
        if debug:
            plt.plot(wavelengths, transmitted_array_stacked.flatten().detach().numpy())
            plt.show()
            print(f'This is the FOM: {FOM}')
        loss_quad = torch.mean(0.5 * (grid_permittivity - 0.5) ** 2)
        fom_loss = -1*(FOM + loss_quad * lambda_quad * iteration ** 1.5)

        if verbose >= 1:
            print(
                f'Losses: {loss_quad * lambda_quad * iteration ** 1.5}, {FOM}')

        fom_loss.backward(retain_graph=True)
        grad_FOM_wrt_transmitted_array = transmitted_array_stacked.grad
        grad_FOM_wrt_grid_permittivity = torch.mul(
            grad_FOM_wrt_transmitted_array, gradient_array_stacked)
        total_gradient = torch.mean(grad_FOM_wrt_grid_permittivity, dim=(0))
        total_gradient = total_gradient.unsqueeze(
            1).repeat(1, useless_transverse_points)

        total_gradients.append(total_gradient)

        if verbose >= 1:
            print(f'Gradient: {torch.mean(torch.abs(total_gradient))}',
                  f'Update: {torch.mean(total_gradient) * learning_rate}')
            print(f'Transmission: {torch.mean(transmitted_array_stacked).item()}',
                  f'Figure-of-Merit: {1-FOM.item()}')
        if debug:
            input("Image analyzed, press enter to continue...")
    grid_permittivity.backward(gradient=torch.stack(total_gradients).mean(
        dim=0), retain_graph=True)
    optimizer.step()
