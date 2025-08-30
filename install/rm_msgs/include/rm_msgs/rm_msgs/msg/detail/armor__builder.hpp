// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from rm_msgs:msg/Armor.idl
// generated code does not contain a copyright notice

#ifndef RM_MSGS__MSG__DETAIL__ARMOR__BUILDER_HPP_
#define RM_MSGS__MSG__DETAIL__ARMOR__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "rm_msgs/msg/detail/armor__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace rm_msgs
{

namespace msg
{

namespace builder
{

class Init_Armor_is_repeat
{
public:
  explicit Init_Armor_is_repeat(::rm_msgs::msg::Armor & msg)
  : msg_(msg)
  {}
  ::rm_msgs::msg::Armor is_repeat(::rm_msgs::msg::Armor::_is_repeat_type arg)
  {
    msg_.is_repeat = std::move(arg);
    return std::move(msg_);
  }

private:
  ::rm_msgs::msg::Armor msg_;
};

class Init_Armor_pose
{
public:
  explicit Init_Armor_pose(::rm_msgs::msg::Armor & msg)
  : msg_(msg)
  {}
  Init_Armor_is_repeat pose(::rm_msgs::msg::Armor::_pose_type arg)
  {
    msg_.pose = std::move(arg);
    return Init_Armor_is_repeat(msg_);
  }

private:
  ::rm_msgs::msg::Armor msg_;
};

class Init_Armor_c_to_a_pitch
{
public:
  explicit Init_Armor_c_to_a_pitch(::rm_msgs::msg::Armor & msg)
  : msg_(msg)
  {}
  Init_Armor_pose c_to_a_pitch(::rm_msgs::msg::Armor::_c_to_a_pitch_type arg)
  {
    msg_.c_to_a_pitch = std::move(arg);
    return Init_Armor_pose(msg_);
  }

private:
  ::rm_msgs::msg::Armor msg_;
};

class Init_Armor_distance_to_image_center
{
public:
  explicit Init_Armor_distance_to_image_center(::rm_msgs::msg::Armor & msg)
  : msg_(msg)
  {}
  Init_Armor_c_to_a_pitch distance_to_image_center(::rm_msgs::msg::Armor::_distance_to_image_center_type arg)
  {
    msg_.distance_to_image_center = std::move(arg);
    return Init_Armor_c_to_a_pitch(msg_);
  }

private:
  ::rm_msgs::msg::Armor msg_;
};

class Init_Armor_type
{
public:
  explicit Init_Armor_type(::rm_msgs::msg::Armor & msg)
  : msg_(msg)
  {}
  Init_Armor_distance_to_image_center type(::rm_msgs::msg::Armor::_type_type arg)
  {
    msg_.type = std::move(arg);
    return Init_Armor_distance_to_image_center(msg_);
  }

private:
  ::rm_msgs::msg::Armor msg_;
};

class Init_Armor_name
{
public:
  explicit Init_Armor_name(::rm_msgs::msg::Armor & msg)
  : msg_(msg)
  {}
  Init_Armor_type name(::rm_msgs::msg::Armor::_name_type arg)
  {
    msg_.name = std::move(arg);
    return Init_Armor_type(msg_);
  }

private:
  ::rm_msgs::msg::Armor msg_;
};

class Init_Armor_id
{
public:
  explicit Init_Armor_id(::rm_msgs::msg::Armor & msg)
  : msg_(msg)
  {}
  Init_Armor_name id(::rm_msgs::msg::Armor::_id_type arg)
  {
    msg_.id = std::move(arg);
    return Init_Armor_name(msg_);
  }

private:
  ::rm_msgs::msg::Armor msg_;
};

class Init_Armor_header
{
public:
  Init_Armor_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Armor_id header(::rm_msgs::msg::Armor::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_Armor_id(msg_);
  }

private:
  ::rm_msgs::msg::Armor msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::rm_msgs::msg::Armor>()
{
  return rm_msgs::msg::builder::Init_Armor_header();
}

}  // namespace rm_msgs

#endif  // RM_MSGS__MSG__DETAIL__ARMOR__BUILDER_HPP_
