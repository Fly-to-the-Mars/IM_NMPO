import numpy as np
import scipy.spatial.transform as spt
import rosbag

bag_f = rosbag.Bag('1e_2.bag')

bag_info = bag_f.get_type_and_topic_info()
print(bag_info.topics)

odom = bag_f.read_messages(topics=["/px4/odom"])
tr = bag_f.read_messages(topics=["/track/thrust_rates"])

velocity = []
velocity_body = []
acceleration = []
acceleration_body = []
for topic, msg, t in odom:
    vel =  np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
    velocity.append(np.array([vel[0], vel[1], vel[2]]))
    q = [-msg.pose.pose.orientation.x, -msg.pose.pose.orientation.y, -msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
    R = spt.Rotation.from_quat(q).as_matrix()
    velocity_body.append(np.dot(R, vel))
    if len(velocity) > 1:
        acceleration.append((velocity[-1]-velocity[-2])/0.01)
        acceleration[-1][2] -= 9.8
        acceleration_body.append(np.dot(R, acceleration[-1]))

velocity = np.array(velocity)
velocity_body = np.array(velocity_body)
acceleration = np.array(acceleration)
acceleration_body = np.array(acceleration_body)

thrust = []
for topic, msg, t in tr:
    thrust.append(msg.thrust)

thrust = np.array(thrust)[2:-2]
print(thrust.shape)


bag_f.close()

import matplotlib.pyplot as plt

# plt.plot(-velocity[:,0])
# plt.plot(-velocity_body[2801:,0],acceleration_body[2800:,0])
# plt.plot(acceleration_body[:,1])
# plt.show()

import scipy.optimize as opt

# x y drag coefficient fitting
def func(x, a):
    return a*x #+b*x+0

idx = np.abs(acceleration_body[2800:,1]) < 6

popt, pcov = opt.curve_fit(func, velocity_body[2801:,1][idx], acceleration_body[2800:,1][idx])
print(popt)
plt.plot((velocity_body[2801:,1][idx]), (acceleration_body[2800:,1][idx]))
plt.plot((velocity_body[2801:,1][idx]), func((velocity_body[2801:,1][idx]), *popt))
plt.show()

# z drag coefficient fitting, thrust coefficient fitting
from mpl_toolkits.mplot3d import Axes3D
# plt3d = Axes3D(plt.figure("hh"))
fig = plt.figure()
plt3d = fig.add_subplot(111, projection='3d')
# plt3d = fig.add_axes(Axes3D(fig))
plt3d.set_xlabel('z velocity')
plt3d.set_ylabel('thrust')
plt3d.set_zlabel('z acceleration')
plt3d.scatter(-velocity_body[2801:-3000,2], thrust[2800:-3000] , acceleration_body[2800:-3000,2])
# plt3d.plot_surface(-velocity_body[2801:,2:2], thrust[2800:].reshape((-1,1)) , acceleration_body[2800:,2:2])

def func2(x, a, b):
    return a*x[0] + b*x[1]

popt, pcov = opt.curve_fit(func2, np.array([velocity_body[2801:-3000,2], thrust[2800:-3000]]), acceleration_body[2800:-3000,2])
print(popt)
# plt3d.plot_surface(-velocity_body[2801:,2:2], thrust[2800:].reshape((-1,1)) , func2(np.array([-velocity_body[2801:,2], thrust[2800:]]), *popt).reshape((-1,1)) )
X = np.linspace(velocity_body[2801:-3000,2].min(), velocity_body[2801:-3000,2].max(), 100)
Y = np.linspace(thrust[2800:-3000].min(), thrust[2800:-3000].max(), 100)
X, Y = np.meshgrid(X, Y)
Z = func2(np.array([X, Y]), *popt)
plt3d.plot_surface(X, Y, Z)

plt.show()


# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import numpy as np

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Make data
# u = np.linspace(0, 2 * np.pi, 100)
# v = np.linspace(0, np.pi, 100)
# x = 10 * np.outer(np.cos(u), np.sin(v))
# y = 10 * np.outer(np.sin(u), np.sin(v))
# z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

# # Plot the surface
# #ax.plot_surface(x, y, z, color='b')
# print(x.shape, y.shape, z.shape)
# ax.plot_surface(x, y, z,cmap='rainbow')

# plt.show()

