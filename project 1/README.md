# Project 1 — Robotic Arm Kinematics & Path Planning

### DecodeLabs Robotics & Automation Internship

## 🎯 Objective

Develop a simulation of a 6-DOF robotic arm that moves its end-effector from **Point A** to **Point B**. The system calculates joint angles using **Inverse Kinematics**, generates a smooth trajectory, and safely avoids obstacles during motion.

---

## 🧠 Project Overview

The project is divided into three main stages:

### Forward Kinematics

Calculates the end-effector position from the robot's joint angles using Denavit-Hartenberg (DH) parameters.

### Inverse Kinematics

Computes the joint angles required to reach a target position using a Damped Least Squares numerical solver while respecting joint limits.

### Trajectory Planning

Creates a smooth quintic trajectory between the start and target positions. If the direct path intersects an obstacle, the planner automatically generates a new collision-free path through a safe waypoint.

---

## 📊 Output

After execution, the program:

* Computes joint angles for each waypoint.
* Generates a collision-free trajectory.
* Displays the robotic arm motion in a 3D Matplotlib simulation.
* Saves the final trajectory image in the **output** folder.


## ⚙️ Customization

You can easily modify:

* Start and target positions
* Obstacle location and size
* Robot DH parameters
* Joint limits
* IK solver settings
* Number of trajectory waypoints

---

## 🔄 ROS 2 Integration

This simulation follows the same workflow used in ROS 2 robotics applications. The kinematics, trajectory planning, and collision checking can be integrated with tools such as **MoveIt**, **TF2**, **RViz**, and **Gazebo** for real robotic systems.

---

## 📌 Submission Files

* Python source files
* Generated trajectory image
* Console output
* Brief explanation of extending the project to full pose control and ROS 2 integration
