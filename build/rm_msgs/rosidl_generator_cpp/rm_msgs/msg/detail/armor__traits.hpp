// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from rm_msgs:msg/Armor.idl
// generated code does not contain a copyright notice

#ifndef RM_MSGS__MSG__DETAIL__ARMOR__TRAITS_HPP_
#define RM_MSGS__MSG__DETAIL__ARMOR__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "rm_msgs/msg/detail/armor__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'pose'
#include "geometry_msgs/msg/detail/pose__traits.hpp"

namespace rm_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const Armor & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: id
  {
    out << "id: ";
    rosidl_generator_traits::value_to_yaml(msg.id, out);
    out << ", ";
  }

  // member: name
  {
    out << "name: ";
    rosidl_generator_traits::value_to_yaml(msg.name, out);
    out << ", ";
  }

  // member: type
  {
    out << "type: ";
    rosidl_generator_traits::value_to_yaml(msg.type, out);
    out << ", ";
  }

  // member: distance_to_image_center
  {
    out << "distance_to_image_center: ";
    rosidl_generator_traits::value_to_yaml(msg.distance_to_image_center, out);
    out << ", ";
  }

  // member: c_to_a_pitch
  {
    out << "c_to_a_pitch: ";
    rosidl_generator_traits::value_to_yaml(msg.c_to_a_pitch, out);
    out << ", ";
  }

  // member: pose
  {
    out << "pose: ";
    to_flow_style_yaml(msg.pose, out);
    out << ", ";
  }

  // member: is_repeat
  {
    out << "is_repeat: ";
    rosidl_generator_traits::value_to_yaml(msg.is_repeat, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Armor & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "id: ";
    rosidl_generator_traits::value_to_yaml(msg.id, out);
    out << "\n";
  }

  // member: name
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "name: ";
    rosidl_generator_traits::value_to_yaml(msg.name, out);
    out << "\n";
  }

  // member: type
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "type: ";
    rosidl_generator_traits::value_to_yaml(msg.type, out);
    out << "\n";
  }

  // member: distance_to_image_center
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "distance_to_image_center: ";
    rosidl_generator_traits::value_to_yaml(msg.distance_to_image_center, out);
    out << "\n";
  }

  // member: c_to_a_pitch
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "c_to_a_pitch: ";
    rosidl_generator_traits::value_to_yaml(msg.c_to_a_pitch, out);
    out << "\n";
  }

  // member: pose
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pose:\n";
    to_block_style_yaml(msg.pose, out, indentation + 2);
  }

  // member: is_repeat
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "is_repeat: ";
    rosidl_generator_traits::value_to_yaml(msg.is_repeat, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Armor & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace rm_msgs

namespace rosidl_generator_traits
{

[[deprecated("use rm_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const rm_msgs::msg::Armor & msg,
  std::ostream & out, size_t indentation = 0)
{
  rm_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use rm_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const rm_msgs::msg::Armor & msg)
{
  return rm_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<rm_msgs::msg::Armor>()
{
  return "rm_msgs::msg::Armor";
}

template<>
inline const char * name<rm_msgs::msg::Armor>()
{
  return "rm_msgs/msg/Armor";
}

template<>
struct has_fixed_size<rm_msgs::msg::Armor>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<rm_msgs::msg::Armor>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<rm_msgs::msg::Armor>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // RM_MSGS__MSG__DETAIL__ARMOR__TRAITS_HPP_
