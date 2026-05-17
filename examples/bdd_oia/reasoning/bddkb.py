# -*- coding: utf-8 -*-
from ablkit.reasoning import KBBase


class BDDKB(KBBase):
    def __init__(self, pseudo_label_list=None):
        if pseudo_label_list is None:
            pseudo_label_list = [0, 1]
        super().__init__(pseudo_label_list)

    def logic_forward(self, attrs):
        """
        Abduction space
        (0, 1, 0, 0) 610812
        (0, 1, 0, 1) 75012
        (0, 1, 1, 0) 75012
        (0, 1, 1, 1) 9212
        (1, 0, 0, 0) 12996
        (1, 0, 0, 1) 1596
        (1, 0, 1, 0) 1596
        (1, 0, 1, 1) 196
        """
        if len(attrs) != 21:
            raise ValueError(
                f"BDDKB.logic_forward expects exactly 21 concept attributes, got {len(attrs)}."
            )
        (
            green_light,
            follow,
            road_clear,
            red_light,
            traffic_sign,
            car,
            person,
            rider,
            other_obstacle,
            left_lane,
            left_green_light,
            left_follow,
            no_left_lane,
            left_obstacle,
            left_solid_line,
            right_lane,
            right_green_light,
            right_follow,
            no_right_lane,
            right_obstacle,
            right_solid_line,
        ) = attrs

        illegal_return = (0, 0, 0, 0)
        if red_light == green_light == 1:
            return illegal_return
        obstacle = car or person or rider or other_obstacle
        if road_clear == obstacle:
            return illegal_return
        move_forward = green_light or follow or road_clear
        stop = red_light or traffic_sign or obstacle
        if stop:
            move_forward = 0

        can_turn_left = left_lane or left_green_light or left_follow
        cannot_turn_left = no_left_lane or left_obstacle or left_solid_line
        turn_left = can_turn_left and int(not cannot_turn_left)

        can_turn_right = right_lane or right_green_light or right_follow
        cannot_turn_right = no_right_lane or right_obstacle or right_solid_line
        turn_right = can_turn_right and int(not cannot_turn_right)

        return move_forward, stop, turn_left, turn_right
