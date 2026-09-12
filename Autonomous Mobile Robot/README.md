# Project 3

### DecodeLabs Robotics & Automation Internship

## 🎯 Objective

Create a simulated autonomous mobile robot that explores a maze, builds a map using LiDAR data, plans a path to its destination, and automatically finds a new route whenever an unexpected obstacle blocks its way.

---

## 🧠 Project Overview

The navigation system is divided into three main components:

### LiDAR Mapping

A simulated 360° LiDAR scans the surroundings and updates a 2D occupancy grid, allowing the robot to gradually build a map of the environment.

### Path Planning

The robot uses the A* algorithm to calculate the shortest path from the start position to the goal based on the information collected from the occupancy grid.

### Dynamic Replanning

If a new obstacle appears during navigation, the robot performs another LiDAR scan, updates its map, and generates a new path to safely continue toward the destination.

---

## 📊 Output

After execution, the program:

* Generates an occupancy grid from LiDAR scans.
* Finds the shortest navigation path.
* Detects dynamic obstacles and replans automatically.
* Produces a visualization of the robot's map and traveled path in the **output** folder.

## ⚙️ Customization

You can modify:

* Maze size
* Start and goal positions
* LiDAR range and resolution
* Dynamic obstacle location
* Path planning parameters

---

## 🤖 Real-World Application

The same navigation principles are widely used in autonomous mobile robots for warehouse automation, indoor delivery, service robotics, and industrial material handling.

---

## 📌 Submission Files

* Python source files
* Generated navigation image
* Console output showing path replanning
* Short explanation of possible future improvements such as higher-resolution LiDAR or alternative path-planning algorithms
