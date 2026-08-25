#pragma once

#include <vector>
#include <limits>

struct Seq {
    int r_index = 0;
    int start_point = 0;
    int end_point = 0;

    Seq() {}
    Seq(int r, int s, int e) : r_index(r), start_point(s), end_point(e) {}
};

struct Move {
    int r_indice[2] = {-2, -2};
    Seq seqList_1[4];
    int len_1 = 0;
    Seq seqList_2[4];
    int len_2 = 0;
    double delta_cost = std::numeric_limits<double>::infinity();

    void copy_from(const Move& other) {
        r_indice[0] = other.r_indice[0];
        r_indice[1] = other.r_indice[1];
        for (int i = 0; i < 4; ++i) {
            seqList_1[i] = other.seqList_1[i];
            seqList_2[i] = other.seqList_2[i];
        }
        len_1 = other.len_1;
        len_2 = other.len_2;
        delta_cost = other.delta_cost;
    }

    Move clone() const {
        Move new_move = *this;
        return new_move;
    }
};
