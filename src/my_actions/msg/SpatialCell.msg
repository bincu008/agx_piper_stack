# Derived immutable layout; only last_scanned is retained per cell.
string id
geometry_msgs/Point minimum
geometry_msgs/Point maximum
geometry_msgs/Point center
# Zero means never observed; ROS observation time, not query time.
builtin_interfaces/Time last_scanned
