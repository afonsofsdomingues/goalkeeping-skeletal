"""
Visualization utils
"""

import numpy as np
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def align_points_and_return_xy(pos):
    X_OFFSET = 52.5
    Y_OFFSET = 34
    x, y = pos[:, 0], pos[:, 1]
    x += X_OFFSET
    y += Y_OFFSET
    return x, y

def plot_xy_track_2d(x, y):
    vp = VerticalPitch(pitch_type='custom', pitch_width=68, pitch_length=105, pitch_color='grass', line_color='white')

    fig, ax = vp.draw()

    # Plot the path
    ax.plot(x, y, color='orange', linewidth=3, label='Ball path')

    # Optional: mark start and end
    ax.scatter(x[0], y[0], color='green', s=80, label='Start')
    ax.scatter(x[-1], y[-1], color='red', s=80, label='End')

    ax.legend(loc='upper right')
    plt.show()

def retrieve_goalkeepers_id(players):
    goalkeepers = []
    for player in players:
        if player['role']['name'] == "Goalkeeper":
            goalkeepers.append(player)
    return goalkeepers

def retrieve_tracking_data_for_player(samples, player_heId):
    player_samples = []
    for sample in samples: 
        if sample['personId']['heId'] == player_heId:
            player_samples.append(sample)
    return player_samples

def plot_skeleton_3d(joints, edges):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = [j[0] for j in joints]
    ys = [j[1] for j in joints]
    zs = [j[2] for j in joints]

    ax.scatter(xs, ys, zs)

    for (i, j) in edges:
        ax.plot(
            [joints[i][0], joints[j][0]],
            [joints[i][1], joints[j][1]],
            [joints[i][2], joints[j][2]],
        )

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.set_ylim(-30, 30)

    plt.show()

def animate_skeleton(frames_list, edges):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Calculate Dynamic Limits
    # Flatten all frames to find the global min/max for the whole video
    all_points = []
    for frame in frames_list:
        for point in frame:
            if point is not None:
                all_points.append(point)
    
    all_points = np.array(all_points)
    
    min_x, max_x = all_points[:, 0].min(), all_points[:, 0].max()
    min_y, max_y = all_points[:, 1].min(), all_points[:, 1].max()
    min_z, max_z = all_points[:, 2].min(), all_points[:, 2].max()

    # Find the center of the data
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2
    mid_z = (min_z + max_z) / 2

    # Determine the largest dimension to make the plot cubic (preserves aspect ratio)
    max_range = max(max_x - min_x, max_y - min_y, max_z - min_z) / 2
    
    # Add padding
    padding = max_range * 1.2

    ax.set_xlim(mid_x - padding, mid_x + padding)
    ax.set_ylim(mid_y - padding, mid_y + padding)
    ax.set_zlim(mid_z - padding, mid_z + padding)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Force aspect ratio to be equal so the person doesn't look stretched
    ax.set_box_aspect([1, 1, 1]) 

    # 2. Setup Plot Elements
    scat = ax.scatter([], [], [], c='red', s=15)
    lines = [ax.plot([], [], [], 'k-', linewidth=2)[0] for _ in edges]

    def update(frame_idx):
        joints = frames_list[frame_idx]

        # Filter out None values safely
        clean_joints = [j for j in joints if j is not None]
        
        if not clean_joints: 
            return lines + [scat]

        # Update Scatter (Joints)
        xs = [j[0] for j in clean_joints]
        ys = [j[1] for j in clean_joints]
        zs = [j[2] for j in clean_joints]
        scat._offsets3d = (xs, ys, zs)

        # Update Lines (Bones)
        for line, (start_idx, end_idx) in zip(lines, edges):
            # Check indices exist and are not None
            if (start_idx < len(joints) and end_idx < len(joints) and 
                joints[start_idx] is not None and joints[end_idx] is not None):
                
                p1 = joints[start_idx]
                p2 = joints[end_idx]

                line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                line.set_3d_properties([p1[2], p2[2]])
            else:
                line.set_data([], [])
                line.set_3d_properties([])

        return lines + [scat]
    
    # 3. Create Animation
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=range(len(frames_list)), 
        interval=50, 
        blit=False
    )
    
    return ani