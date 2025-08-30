// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from rm_msgs:msg/Armor.idl
// generated code does not contain a copyright notice
#include "rm_msgs/msg/detail/armor__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `name`
// Member `type`
#include "rosidl_runtime_c/string_functions.h"
// Member `pose`
#include "geometry_msgs/msg/detail/pose__functions.h"

bool
rm_msgs__msg__Armor__init(rm_msgs__msg__Armor * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    rm_msgs__msg__Armor__fini(msg);
    return false;
  }
  // id
  // name
  if (!rosidl_runtime_c__String__init(&msg->name)) {
    rm_msgs__msg__Armor__fini(msg);
    return false;
  }
  // type
  if (!rosidl_runtime_c__String__init(&msg->type)) {
    rm_msgs__msg__Armor__fini(msg);
    return false;
  }
  // distance_to_image_center
  // c_to_a_pitch
  // pose
  if (!geometry_msgs__msg__Pose__init(&msg->pose)) {
    rm_msgs__msg__Armor__fini(msg);
    return false;
  }
  // is_repeat
  return true;
}

void
rm_msgs__msg__Armor__fini(rm_msgs__msg__Armor * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // id
  // name
  rosidl_runtime_c__String__fini(&msg->name);
  // type
  rosidl_runtime_c__String__fini(&msg->type);
  // distance_to_image_center
  // c_to_a_pitch
  // pose
  geometry_msgs__msg__Pose__fini(&msg->pose);
  // is_repeat
}

bool
rm_msgs__msg__Armor__are_equal(const rm_msgs__msg__Armor * lhs, const rm_msgs__msg__Armor * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // id
  if (lhs->id != rhs->id) {
    return false;
  }
  // name
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->name), &(rhs->name)))
  {
    return false;
  }
  // type
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->type), &(rhs->type)))
  {
    return false;
  }
  // distance_to_image_center
  if (lhs->distance_to_image_center != rhs->distance_to_image_center) {
    return false;
  }
  // c_to_a_pitch
  if (lhs->c_to_a_pitch != rhs->c_to_a_pitch) {
    return false;
  }
  // pose
  if (!geometry_msgs__msg__Pose__are_equal(
      &(lhs->pose), &(rhs->pose)))
  {
    return false;
  }
  // is_repeat
  if (lhs->is_repeat != rhs->is_repeat) {
    return false;
  }
  return true;
}

bool
rm_msgs__msg__Armor__copy(
  const rm_msgs__msg__Armor * input,
  rm_msgs__msg__Armor * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // id
  output->id = input->id;
  // name
  if (!rosidl_runtime_c__String__copy(
      &(input->name), &(output->name)))
  {
    return false;
  }
  // type
  if (!rosidl_runtime_c__String__copy(
      &(input->type), &(output->type)))
  {
    return false;
  }
  // distance_to_image_center
  output->distance_to_image_center = input->distance_to_image_center;
  // c_to_a_pitch
  output->c_to_a_pitch = input->c_to_a_pitch;
  // pose
  if (!geometry_msgs__msg__Pose__copy(
      &(input->pose), &(output->pose)))
  {
    return false;
  }
  // is_repeat
  output->is_repeat = input->is_repeat;
  return true;
}

rm_msgs__msg__Armor *
rm_msgs__msg__Armor__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  rm_msgs__msg__Armor * msg = (rm_msgs__msg__Armor *)allocator.allocate(sizeof(rm_msgs__msg__Armor), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(rm_msgs__msg__Armor));
  bool success = rm_msgs__msg__Armor__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
rm_msgs__msg__Armor__destroy(rm_msgs__msg__Armor * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    rm_msgs__msg__Armor__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
rm_msgs__msg__Armor__Sequence__init(rm_msgs__msg__Armor__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  rm_msgs__msg__Armor * data = NULL;

  if (size) {
    data = (rm_msgs__msg__Armor *)allocator.zero_allocate(size, sizeof(rm_msgs__msg__Armor), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = rm_msgs__msg__Armor__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        rm_msgs__msg__Armor__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
rm_msgs__msg__Armor__Sequence__fini(rm_msgs__msg__Armor__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      rm_msgs__msg__Armor__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

rm_msgs__msg__Armor__Sequence *
rm_msgs__msg__Armor__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  rm_msgs__msg__Armor__Sequence * array = (rm_msgs__msg__Armor__Sequence *)allocator.allocate(sizeof(rm_msgs__msg__Armor__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = rm_msgs__msg__Armor__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
rm_msgs__msg__Armor__Sequence__destroy(rm_msgs__msg__Armor__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    rm_msgs__msg__Armor__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
rm_msgs__msg__Armor__Sequence__are_equal(const rm_msgs__msg__Armor__Sequence * lhs, const rm_msgs__msg__Armor__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!rm_msgs__msg__Armor__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
rm_msgs__msg__Armor__Sequence__copy(
  const rm_msgs__msg__Armor__Sequence * input,
  rm_msgs__msg__Armor__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(rm_msgs__msg__Armor);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    rm_msgs__msg__Armor * data =
      (rm_msgs__msg__Armor *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!rm_msgs__msg__Armor__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          rm_msgs__msg__Armor__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!rm_msgs__msg__Armor__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
