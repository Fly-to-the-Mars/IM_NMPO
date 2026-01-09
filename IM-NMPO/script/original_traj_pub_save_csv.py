# def pub_traj(opt_t_res, opt:WayPointOpt,csv_filename=None):
#     x = opt_t_res['x'].full().flatten()
#     traj = TrackTraj()

#     if opt._loop:
#         s = x[(opt._Herizon-1)*opt._X_dim:]
#     else:
#         s = opt._xinit
#     pos = Point()
#     pos.x = s[0]
#     pos.y = s[1]
#     pos.z = s[2]
#     vel = Vector3()
#     vel.x = s[3]
#     vel.y = s[4]
#     vel.z = s[5]
#     quat = Quaternion()
#     quat.w = s[6]
#     quat.x = s[7]
#     quat.y = s[8]
#     quat.z = s[9]
#     angular = Vector3()
#     angular.x = s[10]
#     angular.y = s[11]
#     angular.z = s[12]

#     traj.position.append(pos)
#     traj.velocity.append(vel)
#     traj.orientation.append(quat)
#     traj.angular.append(angular)
        
#     for i in range(opt._wp_num):
#         for j in range(opt._Ns[i]):
#             idx = opt._N_wp_base[i]+j
#             s = x[idx*opt._X_dim: (idx+1)*opt._X_dim]
#             pos = Point()
#             pos.x = s[0]
#             pos.y = s[1]
#             pos.z = s[2]
#             vel = Vector3()
#             vel.x = s[3]
#             vel.y = s[4]
#             vel.z = s[5]
#             quat = Quaternion()
#             quat.w = s[6]
#             quat.x = s[7]
#             quat.y = s[8]
#             quat.z = s[9]
#             angular = Vector3()
#             angular.x = s[10]
#             angular.y = s[11]
#             angular.z = s[12]
#             dt = x[-opt._wp_num+i]

#             traj.position.append(pos)
#             traj.velocity.append(vel)
#             traj.orientation.append(quat)
#             traj.angular.append(angular)
#             traj.dt.append(dt)
#     traj_pub.publish(traj)
    # csv_filename = "/home/kingstrong/IM_NMPO_ws/src/Fast-fly/script/time_optimal_traj.csv"
    # save_traj_from_pub_traj(opt_t_res, opt, csv_filename)
    
# def save_traj_from_pub_traj(res, opt: WayPointOpt, csv_f):
#     """直接从pub_traj的逻辑生成CSV文件"""
#     with open(csv_f, 'w') as f:
#         traj_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         labels = ['t',
#                   "p_x", "p_y", "p_z",
#                   "v_x", "v_y", "v_z",
#                   "q_w", "q_x", "q_y", "q_z",
#                   "w_x", "w_y", "w_z",
#                   "u_1", "u_2", "u_3", "u_4"]
#         traj_writer.writerow(labels)
#         x = res['x'].full().flatten()
        
#         t = 0
        
#         if opt._loop:
#             s = x[(opt._Herizon-1)*opt._X_dim:]
#             u = x[opt._Herizon*opt._X_dim+(opt._Herizon-1)*opt._U_dim: opt._Herizon*opt._X_dim+opt._Herizon*opt._U_dim]
#         else:
#             s = opt._xinit
#             u = x[opt._Herizon*opt._X_dim: opt._Herizon*opt._X_dim+opt._U_dim]
        
#         traj_writer.writerow([t, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], s[12], u[0], u[1], u[2], u[3]])
        
#         for i in range(opt._wp_num):
#             dt = x[-opt._wp_num+i]
#             for j in range(opt._Ns[i]):
#                 idx = opt._N_wp_base[i]+j
#                 t += dt
#                 s = x[idx*opt._X_dim: (idx+1)*opt._X_dim]
#                 if idx != opt._Herizon-1:
#                     u = x[opt._Herizon*opt._X_dim+(idx+1)*opt._U_dim: opt._Herizon*opt._X_dim+(idx+2)*opt._U_dim]
#                 else:
#                     u = [0,0,0,0]
#                 traj_writer.writerow([t, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], s[12], u[0], u[1], u[2], u[3]])

# def pub_path_visualization(opt_t_res, opt:WayPointOpt):
#     x = opt_t_res['x'].full().flatten()

#     msg = Path()
#     msg.header.stamp = rospy.Time.now()
#     msg.header.frame_id = "world"
#     for i in range(opt._Herizon):
#         pos = PoseStamped()
#         pos.header.frame_id = "world"
#         pos.pose.position.y = x[i*opt._X_dim+0]
#         pos.pose.position.x = x[i*opt._X_dim+1]
#         pos.pose.position.z = -x[i*opt._X_dim+2]

#         pos.pose.orientation.w = 1
#         pos.pose.orientation.y = 0
#         pos.pose.orientation.x = 0
#         pos.pose.orientation.z = 0
#         msg.poses.append(pos)
#     planned_path_pub.publish(msg)