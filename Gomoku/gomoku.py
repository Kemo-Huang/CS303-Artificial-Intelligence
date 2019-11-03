import numpy as np
import functools
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0

MY_SLEEP_ONE = 15
RIVAL_SLEEP_ONE = 10
RIVAL_LIVE_ONE = 26
MY_LIVE_ONE = 42
RIVAL_SLEEP_TWO = 75
MY_SLEEP_TWO = 120
RIVAL_JUMP_LIVE_TWO = 240
RIVAL_BIG_JUMP_LIVE_TWO = 240
MY_JUMP_LIVE_TWO = 260
MY_BIG_JUMP_LIVE_TWO = 260
RIVAL_LIVE_TWO = 415
MY_LIVE_TWO = 450
RIVAL_SLEEP_THREE = 500
MY_SLEEP_THREE = 650
RIVAL_JUMP_LIVE_THREE = 1150
RIVAL_LIVE_THREE = 1425
MY_JUMP_LIVE_THREE = 1550
MY_LIVE_THREE = 1730
RIVAL_FLUSH_FOUR = 1750
MY_FLUSH_FOUR = 2450
RIVAL_LIVE_FOUR = 3600
MY_LIVE_FOUR = 4750
MY_FIVE = 200000
RIVAL_FIVE = 150000

MAX = MY_FIVE * 100
MIN = -MAX
heuristic_max_len = 10
math_threshold = 1.4
check_depth = 2
search_depth = 4
extend_limit = 1

LOC_IDX = 0
MS_IDX = 1
RS_IDX = 2
COLOR_IDX = 3
SCORE_IDX = 4
STEP_IDX = 5
MAX_SCORE = 0
MIN_SCORE = 0
last_max_point = None
last_min_point = None


class AI(object):

    def __init__(self, board_size, color, time_out):
        self.board_size = board_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.chessboard = None
        self.my_piece = None
        self.my_max_score = 0
        self.rival_max_score = 0
        self.count = 0 if color == COLOR_BLACK else 1
        self.my_score = np.zeros((board_size, board_size), dtype=np.int)
        self.rival_score = np.zeros((board_size, board_size), dtype=np.int)
        self.score_cache = [
            [
                np.zeros((board_size, board_size), dtype=np.int),
                np.zeros((board_size, board_size), dtype=np.int),
                np.zeros((board_size, board_size), dtype=np.int),
                np.zeros((board_size, board_size), dtype=np.int)
            ],
            [
                np.zeros((board_size, board_size), dtype=np.int),
                np.zeros((board_size, board_size), dtype=np.int),
                np.zeros((board_size, board_size), dtype=np.int),
                np.zeros((board_size, board_size), dtype=np.int)
            ]
        ]
        global center
        center = int(self.board_size / 2)

    def go(self, board):
        start = time.time()
        self.chessboard = board
        self.candidate_list.clear()
        pieces = np.where(board != COLOR_NONE)
        pieces = list(zip(pieces[0], pieces[1]))
        cnt = len(pieces)
        if cnt > 1:
            self.init_score()
            self.alphabeta(search_depth, MIN, MAX, self.color, 1, 0)
            # self.candidate_list.sort(key=functools.cmp_to_key(cmp_score_1))
            print('sorted:', self.candidate_list)
            if not self.candidate_list:
                px = self.my_piece[0]
                py = self.my_piece[1]
                if px > center:
                    self.candidate_list = [(px - 1, py)]
                else:
                    self.candidate_list = [(px + 1, py)]
            else:
                self.candidate_list = [self.candidate_list[-1]]
        elif cnt == 0:
            idx = (center, center)
            self.candidate_list = [idx]
        else:
            pieces = pieces[0]
            if pieces[0] > center:
                if pieces[1] > center:
                    idx = (pieces[0] - 1, pieces[1] - 1)
                else:
                    idx = (pieces[0] - 1, pieces[1] + 1)
            else:
                if pieces[1] > center:
                    idx = (pieces[0] + 1, pieces[1] - 1)
                else:
                    idx = (pieces[0] + 1, pieces[1] + 1)
            self.candidate_list = [idx]
        self.my_piece = self.candidate_list[-1]
        self.count += 2
        print(time.time() - start)

    def init_score(self):
        idx = np.where(self.chessboard == COLOR_NONE)
        available_pieces = list(zip(idx[0], idx[1]))
        for piece in available_pieces:
            i = piece[0]
            j = piece[1]
            field = self.extract_neighbor_field(piece, 2)
            neighbors = np.where(field != COLOR_NONE)
            if len(neighbors[0]) >= 2:
                self.my_score[i, j] = self.evaluate(i, j, True, None)
                self.rival_score[i, j] = self.evaluate(i, j, False, None)

    def update_on_line(self, px, py, direction):
        p = self.chessboard[px, py]
        if p == COLOR_NONE:
            self.my_score[px, py] = self.evaluate(px, py, True, direction)
            self.rival_score[px, py] = self.evaluate(px, py, False, direction)
        else:
            self.my_score[px, py] = 0
            self.rival_score[px, py] = 0

    def update_score(self, piece):
        dis = 4
        size = self.board_size
        for i in range(-dis, dis + 1):
            x = piece[0]
            y = piece[1] + i
            if y < 0:
                continue
            if y >= size:
                break
            self.update_on_line(x, y, 0)
        for i in range(-dis, dis + 1):
            x = piece[0] + i
            y = piece[1]
            if x < 0:
                continue
            if x >= size:
                break
            self.update_on_line(x, y, 1)
        for i in range(-dis, dis + 1):
            x = piece[0] + i
            y = piece[1] + i
            if x < 0 or y < 0:
                continue
            if x >= size or y >= size:
                break
            self.update_on_line(x, y, 2)
        for i in range(-dis, dis + 1):
            x = piece[0] + i
            y = piece[1] - i
            if x < 0 or y >= size:
                continue
            if x >= size or y < 0:
                break
            self.update_on_line(x, y, 3)

    def move_piece(self, piece, color):
        idx = piece[LOC_IDX]
        self.chessboard[idx[0], idx[1]] = color
        self.update_score(idx)
        self.count += 1

    def remove_piece(self, piece):
        idx = piece[LOC_IDX]
        self.chessboard[idx[0], idx[1]] = COLOR_NONE
        self.update_score(idx)
        self.count -= 1

    def extract_neighbor_field(self, chess, dis):
        x = chess[0]
        y = chess[1]
        board = self.chessboard
        sx = x - dis if x > dis else 0
        ex = x + dis + 1 if x + dis + 1 < self.board_size else self.board_size
        sy = y - dis if y > dis else 0
        ey = y + dis + 1 if y + dis + 1 < self.board_size else self.board_size
        field = board[sx:ex, sy:ey]
        return field

    def evaluate(self, px, py, is_self, direction):
        size = self.board_size
        cache_idx = 0 if is_self else 1
        color = self.color if is_self else -self.color
        # -
        count_1 = 1
        count_2 = 1
        empty_loc = 0
        end_1 = 0
        end_2 = 0
        empty_two = False
        if direction is None or direction == 0:
            i = py + 1
            limit = min(py + 5, size)
            while i < py + 5 and count_1 < 5:
                if i >= size:
                    end_1 = 1
                    break
                piece = self.chessboard[px, i]
                if piece == COLOR_NONE:
                    if i + 1 < limit and empty_loc == 0:
                        next_piece = self.chessboard[px, i + 1]
                        if next_piece == color:
                            empty_loc = count_1
                            count_1 += 1
                            i += 2
                            continue
                        if next_piece == COLOR_NONE and i + 2 < limit \
                                and count_1 == 1 and self.chessboard[px, i + 2] == color:
                            count_1 += 1
                            empty_two = True
                        break
                    break
                if piece == color:
                    count_1 += 1
                    i += 1
                    continue
                end_1 = 1
                break

            if not empty_two:
                i = py - 1
                limit = max(py - 5, 0)
                while i > py - 5 and count_2 < 5:
                    if i < 0:
                        end_2 = 1
                        break
                    piece = self.chessboard[px, i]
                    if piece == COLOR_NONE:
                        if i - 1 > limit and empty_loc == 0:
                            next_piece = self.chessboard[px, i - 1]
                            if next_piece == color:
                                empty_loc = -count_2
                                count_2 += 1
                                i -= 2
                                continue
                            if next_piece == COLOR_NONE and i - 2 > limit \
                                    and count_2 == 1 and end_1 == 0 and self.chessboard[px, i - 2] == color:
                                count_2 += 1
                                empty_two = True
                            break
                        break
                    if piece == color:
                        count_2 += 1
                        i -= 1
                        continue
                    end_2 = 1
                    break

            self.score_cache[cache_idx][0][px, py] = self.count_to_score(color, count_1, count_2,
                                                                         empty_two, empty_loc, end_1, end_2)
        # |
        count_1 = 1
        count_2 = 1

        empty_loc = 0
        end_1 = 0
        end_2 = 0
        empty_two = False
        if direction is None or direction == 1:
            i = px + 1
            limit = min(px + 5, size)
            while i < px + 5 and count_1 < 5:
                if i >= size:
                    end_1 = 1
                    break
                piece = self.chessboard[i, py]
                if piece == COLOR_NONE:
                    if i + 1 < limit and empty_loc == 0:
                        next_piece = self.chessboard[i + 1, py]
                        if next_piece == color:
                            empty_loc = count_1
                            count_1 += 1
                            i += 2
                            continue
                        if next_piece == COLOR_NONE and i + 2 < limit \
                                and count_1 == 1 and self.chessboard[i + 2, py] == color:
                            count_1 += 1
                            empty_two = True
                        break
                    break
                if piece == color:
                    count_1 += 1
                    i += 1
                    continue
                end_1 = 1
                break

            if not empty_two:
                i = px - 1
                limit = max(px - 5, 0)
                while i > px - 5 and count_2 < 5:
                    if i < 0:
                        end_2 = 1
                        break
                    piece = self.chessboard[i, py]
                    if piece == COLOR_NONE:
                        if i - 1 > limit and empty_loc == 0:
                            next_piece = self.chessboard[i - 1, py]
                            if next_piece == color:
                                empty_loc = -count_2
                                count_2 += 1
                                i -= 2
                                continue
                            if next_piece == COLOR_NONE and i - 2 > limit \
                                    and count_2 == 1 and end_1 == 0 and self.chessboard[i - 2, py] == color:
                                count_2 += 1
                                empty_two = True
                            break
                        break
                    if piece == color:
                        count_2 += 1
                        i -= 1
                        continue

                    end_2 = 1
                    break
            self.score_cache[cache_idx][1][px, py] = self.count_to_score(color, count_1, count_2,
                                                                         empty_two, empty_loc, end_1, end_2)

        # \
        count_1 = 1
        count_2 = 1
        empty_loc = 0
        end_1 = 0
        end_2 = 0
        empty_two = False
        if direction is None or direction == 2:
            k = 1
            lim_x = min(px + 5, size)
            lim_y = min(py + 5, size)
            while k < 5 and count_1 < 5:
                i = px + k
                j = py + k
                if i >= size or j >= size:
                    end_1 = 1
                    break
                piece = self.chessboard[i, j]
                if piece == COLOR_NONE:
                    if i + 1 < lim_x and j + 1 < lim_y and empty_loc == 0:
                        next_piece = self.chessboard[i + 1, j + 1]
                        if next_piece == color:
                            empty_loc = count_1
                            count_1 += 1
                            k += 2
                            continue
                        if next_piece == COLOR_NONE and i + 2 < lim_x \
                                and j + 2 < lim_y and count_1 == 1 \
                                and self.chessboard[i + 2, j + 2] == color:
                            count_1 += 1
                            empty_two = True
                        break
                    break
                if piece == color:
                    count_1 += 1
                    k += 1
                    continue
                end_1 = 1
                break

            if not empty_two:
                k = 1
                lim_x = max(px - 5, 0)
                lim_y = max(py - 5, 0)
                while k < 5 and count_2 < 5:
                    i = px - k
                    j = py - k
                    if i < 0 or j < 0:
                        end_2 = 1
                        break
                    piece = self.chessboard[i, j]
                    if piece == COLOR_NONE:
                        if i - 1 > lim_x and j - 1 > lim_y and empty_loc == 0:
                            next_piece = self.chessboard[i - 1, j - 1]
                            if next_piece == color:
                                empty_loc = -count_2
                                count_2 += 1
                                k += 2
                                continue
                            if next_piece == COLOR_NONE and i - 2 > lim_x \
                                    and j - 2 > lim_y and count_2 == 1 and end_1 == 0 \
                                    and self.chessboard[i - 2, j - 2] == color:
                                count_2 += 1
                                empty_two = True
                            break
                        break
                    if piece == color:
                        count_2 += 1
                        k += 1
                        continue
                    end_2 = 1
                    break
            self.score_cache[cache_idx][2][px, py] = self.count_to_score(color, count_1, count_2,
                                                                         empty_two, empty_loc, end_1, end_2)
        # /
        count_1 = 1
        count_2 = 1
        empty_loc = 0
        end_1 = 0
        end_2 = 0
        empty_two = False
        if direction is None or direction == 3:
            k = 1
            lim_x = min(px + 5, size)
            lim_y = max(py - 5, 0)
            while k < 5 and count_1 < 5:
                i = px + k
                j = py - k
                if i >= size or j < 0:
                    end_1 = 1
                    break
                piece = self.chessboard[i, j]
                if piece == COLOR_NONE:
                    if i + 1 < lim_x and j - 1 > lim_y and empty_loc == 0:
                        next_piece = self.chessboard[i + 1, j - 1]
                        if next_piece == color:
                            empty_loc = count_1
                            count_1 += 1
                            k += 2
                            continue
                        if next_piece == COLOR_NONE and i + 2 < lim_x \
                                and j - 2 > lim_y and count_1 == 1 \
                                and self.chessboard[i + 2, j - 2] == color:
                            count_1 += 1
                            empty_two = True
                        break
                    break
                if piece == color:
                    count_1 += 1
                    k += 1
                    continue
                end_1 = 1
                break

            if not empty_two:
                k = 1
                lim_x = max(px - 5, 0)
                lim_y = min(py + 5, size)
                while k < 5 and count_2 < 5:
                    i = px - k
                    j = py + k
                    if i < 0 or j >= size:
                        end_2 = 1

                        break
                    piece = self.chessboard[i, j]
                    if piece == COLOR_NONE:
                        if i - 1 > lim_x and j + 1 < lim_y and empty_loc == 0:
                            next_piece = self.chessboard[i - 1, j + 1]
                            if next_piece == color:
                                empty_loc = -count_2
                                count_2 += 1
                                k += 2
                                continue
                            if next_piece == COLOR_NONE and i - 2 > lim_x \
                                    and j + 2 < lim_y and count_2 == 1 and end_1 == 0 \
                                    and self.chessboard[i - 2, j + 2] == color:
                                count_2 += 1
                                empty_two = True
                            break
                        break
                    if piece == color:
                        count_2 += 1
                        k += 1
                        continue

                    end_2 = 1
                    break

            self.score_cache[cache_idx][3][px, py] = self.count_to_score(color, count_1, count_2,
                                                                         empty_two, empty_loc, end_1, end_2)
        return self.score_cache[cache_idx][0][px, py] + \
               self.score_cache[cache_idx][1][px, py] + \
               self.score_cache[cache_idx][2][px, py] + \
               self.score_cache[cache_idx][3][px, py]

    def count_to_score(self, color, count_1, count_2, empty_two, empty_loc, end_1, end_2):
        count = count_1 + count_2 - 1
        if empty_two:
            if color == self.color:
                return MY_BIG_JUMP_LIVE_TWO
            else:
                return RIVAL_BIG_JUMP_LIVE_TWO
        block = end_1 + end_2
        if count < 5:
            if count == 1:
                if block > 1:
                    return 0
                if block == 1:
                    if color == self.color:
                        return MY_SLEEP_ONE
                    else:
                        return RIVAL_SLEEP_ONE
                if color == self.color:
                    return MY_LIVE_ONE
                else:
                    return RIVAL_LIVE_ONE
            if count == 2:
                if block > 1:
                    return 0
                if block == 1:
                    if color == self.color:
                        return MY_SLEEP_TWO
                    else:
                        return RIVAL_SLEEP_TWO
                if empty_loc != 0:
                    if color == self.color:
                        return MY_JUMP_LIVE_TWO
                    else:
                        return RIVAL_JUMP_LIVE_TWO
                if color == self.color:
                    return MY_LIVE_TWO
                else:
                    return RIVAL_LIVE_TWO
            if count == 3:
                if block > 1:
                    return 0
                if block == 1:
                    if color == self.color:
                        return MY_SLEEP_THREE
                    else:
                        return RIVAL_SLEEP_THREE
                if empty_loc != 0:
                    if color == self.color:
                        return MY_JUMP_LIVE_THREE
                    else:
                        return RIVAL_JUMP_LIVE_THREE
                if color == self.color:
                    return MY_LIVE_THREE
                else:
                    return RIVAL_LIVE_THREE
            if count == 4:
                if block > 2:
                    return 0
                if empty_loc != 0:
                    if color == self.color:
                        return MY_FLUSH_FOUR
                    else:
                        return RIVAL_FLUSH_FOUR
                elif block == 2:
                    return 0
                elif block == 1:
                    if color == self.color:
                        return MY_FLUSH_FOUR
                    else:
                        return RIVAL_FLUSH_FOUR
                if color == self.color:
                    return MY_LIVE_FOUR
                else:
                    return RIVAL_LIVE_FOUR
        else:
            if empty_loc == 0:
                if color == self.color:
                    return MY_FIVE
                else:
                    return RIVAL_FIVE
            else:
                if empty_loc > 0:
                    # space is in count_1
                    connected_2 = count_2 + empty_loc - 1
                    if connected_2 >= 5:
                        if color == self.color:
                            return MY_FIVE
                        else:
                            return RIVAL_FIVE
                    if connected_2 == 4 and end_2 == 0:
                        if color == self.color:
                            return MY_LIVE_FOUR
                        else:
                            return RIVAL_LIVE_FOUR
                    if color == self.color:
                        return MY_FLUSH_FOUR
                    else:
                        return RIVAL_FLUSH_FOUR
                else:
                    # space is in count_2
                    connected_1 = count_1 - empty_loc - 1
                    if connected_1 >= 5:
                        if color == self.color:
                            return MY_FIVE
                        else:
                            return RIVAL_FIVE
                    if connected_1 == 4 and end_1 == 0:
                        if color == self.color:
                            return MY_LIVE_FOUR
                        else:
                            return RIVAL_LIVE_FOUR
                    if color == self.color:
                        return MY_FLUSH_FOUR
                    else:
                        return RIVAL_FLUSH_FOUR

    def heuristic(self, color):
        fives = []
        my_live_fours = []
        rival_live_fours = []
        my_flush_fours = []
        rival_flush_fours = []
        my_two_threes = []
        rival_two_threes = []
        my_threes = []
        rival_threes = []
        my_twos = []
        rival_twos = []
        ones = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.chessboard[i, j] != COLOR_NONE:
                    continue
                # discard the sparse pieces
                field = self.extract_neighbor_field((i, j), 2)
                if len(np.where(field != COLOR_NONE)[0]) < 2:
                    continue
                # begin
                rs = self.rival_score[i, j]
                ms = self.my_score[i, j]
                max_score = max(ms, rs)
                ###################################################################
                p = [(i, j), ms, rs, color, max_score, 0]
                ###################################################################
                if ms >= MY_FIVE:
                    fives.append(p)
                elif rs >= RIVAL_FIVE:
                    fives.append(p)
                elif ms >= MY_LIVE_FOUR:
                    my_live_fours.append(p)
                elif rs >= RIVAL_LIVE_FOUR:
                    rival_live_fours.append(p)
                elif ms >= MY_FLUSH_FOUR:
                    my_flush_fours.append(p)
                elif rs >= RIVAL_FLUSH_FOUR:
                    rival_flush_fours.append(p)
                elif ms >= 2 * MY_JUMP_LIVE_THREE:
                    my_two_threes.append(p)
                elif rs >= 2 * RIVAL_JUMP_LIVE_THREE:
                    rival_two_threes.append(p)
                elif ms >= MY_JUMP_LIVE_THREE:
                    my_threes.append(p)
                elif rs >= RIVAL_JUMP_LIVE_THREE:
                    rival_threes.append(p)
                elif ms >= MY_SLEEP_TWO:
                    my_twos.insert(0, p)
                elif rs >= RIVAL_SLEEP_TWO:
                    rival_twos.insert(0, p)
                else:
                    ones.append(p)
        if fives:
            fives.sort(key=lambda x: x[SCORE_IDX], reverse=True)
            return fives
        if color == self.color and my_live_fours:
            my_live_fours.sort(key=lambda x: x[SCORE_IDX], reverse=True)
            return my_live_fours
        if color == -self.color and rival_live_fours:
            rival_live_fours.sort(key=lambda x: x[SCORE_IDX], reverse=True)
            return rival_live_fours
        if color == self.color and rival_live_fours and not my_flush_fours:
            rival_live_fours.sort(key=lambda x: x[SCORE_IDX], reverse=True)
            return rival_live_fours
        if color == -self.color and my_live_fours and not rival_flush_fours:
            my_live_fours.sort(key=lambda x: x[SCORE_IDX], reverse=True)
            return my_live_fours

        fours = my_live_fours + rival_live_fours if color == self.color \
            else rival_live_fours + my_live_fours
        flush_fours = my_flush_fours + rival_flush_fours if color == self.color \
            else rival_flush_fours + my_flush_fours
        if fours:
            result = fours + flush_fours
            result.sort(key=lambda x: x[SCORE_IDX], reverse=True)
            return result

        if color == self.color:
            result = my_two_threes + rival_two_threes + my_flush_fours \
                     + rival_flush_fours + my_threes + rival_threes
        else:
            result = rival_two_threes + my_two_threes + rival_flush_fours \
                     + my_flush_fours + rival_threes + my_threes
        if color == self.color:
            twos = my_twos + rival_twos
        else:
            twos = rival_twos + my_twos
        result += twos if twos else ones
        result.sort(key=lambda x: x[SCORE_IDX], reverse=True)
        if len(result) > heuristic_max_len:
            return result[: heuristic_max_len]
        return result

    # def negamax(self):
    #     alpha = MIN
    #     beta = MAX
    #     for p in self.candidate_list:
    #         self.move_piece(p, self.color)
    #         v = self.alphabeta(search_depth - 1, alpha, beta, -self.color, 1, 0)
    #         v[0] *= -1
    #         alpha = max(alpha, v[0])
    #         self.remove_piece(p)
    #         p[SCORE_IDX] = v[0]
    #         p[STEP_IDX] = v[1]
    #     return alpha

    def alphabeta(self, depth, alpha, beta, color, step, extend):
        pieces = self.heuristic(color)
        print(pieces)
        # print(len(pieces), pieces)
        if depth == 0 or not pieces or pieces[0][SCORE_IDX] >= RIVAL_FIVE:
            self.my_max_score = 0
            self.rival_max_score = 0
            empty_idx = np.where(self.chessboard == COLOR_NONE)
            empty_pieces = list(zip(empty_idx[0], empty_idx[1]))
            for p in empty_pieces:
                self.my_max_score = max(self.my_max_score, self.my_score[p[0], p[1]])
                self.rival_max_score = max(self.rival_max_score, self.rival_score[p[0], p[1]])
            if color == self.color:
                eva = self.my_max_score - self.rival_max_score
            else:
                eva = self.rival_max_score - self.my_max_score
            return [eva, step]

        # if RIVAL_LIVE_FOUR > eva > -RIVAL_LIVE_FOUR:
        #     _score, _len = self.find_check(color, depth, True)
        #     if _score:
        #         return (_score, step + _len)
        #
        # if RIVAL_JUMP_LIVE_THREE * 2 > eva > RIVAL_JUMP_LIVE_THREE * (-2):
        #     _score, _len = self.find_check(color, depth, False)
        #     if _score:
        #         return (_score, step + _len)
        #
        # best = [MIN, step]
        for piece in pieces:
            self.move_piece(piece, color)
            print(piece)
            print(self.chessboard)
            _depth = depth - 1
            # if extend < extend_limit:
            #     if (color == self.color and piece[MS_IDX] >= FIVE) or (
            #             color == -self.color and piece[RS_IDX] >= FIVE):
            #         _depth += 2
            #         extend += 1
            #     if (color == self.color and piece[MS_IDX] >= MY_JUMP_LIVE_THREE * 2) or (
            #             color == -self.color and piece[RS_IDX] >= RIVAL_JUMP_LIVE_THREE * 2):
            #         _depth = depth
            #         extend += 1
            v = self.alphabeta(_depth, -beta, -alpha, -color, step + 1, extend)
            v[0] *= -1
            best = [MIN, step]
            self.remove_piece(piece)
            if v[0] > best[0]:
                best = v
            if best[0] > alpha:
                alpha = best[0]
                if depth == search_depth:
                    self.candidate_list = [piece[LOC_IDX]]
            if greater(alpha, beta):
                return [beta, step]
        return [alpha, step]

    # def find_greater_than(self, color, _score):
    #     result = []
    #     fives = []
    #     idx = np.where(self.chessboard == COLOR_NONE)
    #     for piece in list(zip(idx[0], idx[1])):
    #         p = {
    #             'idx': piece,
    #             'score': 0
    #         }
    #         if self.my_score[piece[0], piece[1]] >= FIVE:
    #             p['score'] = FIVE
    #             if color == -self.color:
    #                 p['score'] *= -1
    #             fives.append(p)
    #         elif self.rival_score[piece[0], piece[1]] >= FIVE:
    #             p['score'] = FIVE
    #             if color == self.color:
    #                 p['score'] *= -1
    #             fives.append(p)
    #         else:
    #             last_min_idx = None
    #             if last_max_point:
    #                 last_min_idx = last_max_point['idx']
    #             if not last_max_point or piece[0] == last_min_idx[0] or piece[1] == last_min_idx[1] \
    #                     or abs(last_min_idx[0] - piece[0]) == abs(last_min_idx[1] - piece[1]):
    #                 sc = self.my_score[piece[0], piece[1]] if color == self.color else self.rival_score[
    #                     piece[0], piece[1]]
    #                 p['score'] = sc
    #                 if sc >= _score:
    #                     result.append(p)
    #     if len(fives):
    #         return fives
    #     result.sort(reverse=True, key=lambda x: x['score'])
    #     return result
    #
    # def find_less_than(self, color, _score):
    #     result = []
    #     fives = []
    #     fours = []
    #     flush_fours = []
    #     idx = np.where(self.chessboard == COLOR_NONE)
    #     for piece in list(zip(idx[0], idx[1])):
    #         p = [piece, 0]
    #         my_sc = self.my_score[piece[0], piece[1]] if color == self.color else self.rival_score[piece[0], piece[1]]
    #         ri_sc = self.rival_score[piece[0], piece[1]] if color == self.color else self.my_score[piece[0], piece[1]]
    #         if my_sc >= FIVE:
    #             p[1] = -my_sc
    #             return [p]
    #         if ri_sc >= FIVE:
    #             p[1] = ri_sc
    #             fives.append(p)
    #             continue
    #         if my_sc >= MY_LIVE_FOUR:
    #             p[1] = -my_sc
    #             fours.insert(0, p)
    #             continue
    #         if ri_sc >= RIVAL_LIVE_FOUR:
    #             fours.append(p)
    #             continue
    #         if my_sc >= MY_FLUSH_FOUR:
    #             p[1] = -my_sc
    #             flush_fours.insert(0, p)
    #             continue
    #         if ri_sc >= RIVAL_FLUSH_FOUR:
    #             p[1] = ri_sc
    #             flush_fours.append(p)
    #             continue
    #         if my_sc >= _score or ri_sc >= _score:
    #             p[1] = my_sc
    #             result.append(p)
    #     if fives:
    #         return fives
    #     if fours:
    #         return fours + flush_fours
    #     result = flush_fours + result
    #     result.sort(reverse=True, key=lambda x: abs(x[1]))
    #     return result

    # def _max(self, color, depth):
    #     global last_max_point
    #     if depth <= 1:
    #         return None
    #     pieces = self.find_greater_than(color, MAX_SCORE)
    #     if not len(pieces):
    #         return None
    #     if pieces[0]['score'] >= FOUR:
    #         return [pieces[0]]
    #     for p in pieces:
    #         self.move_piece(p, color)
    #         if p['score'] > -FIVE:
    #             last_max_point = p
    #         result = self._min(-color, depth - 1)
    #         self.remove_piece(p)
    #         if result is not None:
    #             result.insert(0, p)
    #             return result
    #         else:
    #             return [p]
    #     return None
    #
    # def _min(self, color, depth):
    #     # winner = self.is_win()
    #     # if winner:
    #     #     return None if winner == color else []
    #     global last_min_point
    #     if depth <= 1:
    #         return None
    #     pieces = self.find_less_than(color, MIN_SCORE)
    #     if not len(pieces):
    #         return None
    #     if -pieces[0]['score'] >= FOUR:
    #         return None
    #     candidates = []
    #     for p in pieces:
    #         self.move_piece(p, color)
    #         last_min_point = p
    #         result = self._max(-color, depth - 1)
    #         self.remove_piece(p)
    #         if result is not None:
    #             result.insert(0, p)
    #             candidates.append(result)
    #             continue
    #         else:
    #             return None
    #     return candidates[randint(0, len(candidates) - 1)]

    def is_win(self):
        board = self.chessboard
        size = self.board_size
        for i in range(size):
            for j in range(size):
                piece = board[i, j]
                if piece == COLOR_NONE:
                    continue
                cnt = 1
                k = j + 1
                while True:
                    if k >= size:
                        break
                    if board[i, k] != piece:
                        break
                    cnt += 1
                    k += 1
                k = j - 1
                while True:
                    if k < 0:
                        break
                    if board[i, k] != piece:
                        break
                    cnt += 1
                    k -= 1
                if cnt >= 5:
                    return piece

                cnt = 1
                k = i + 1
                while True:
                    if k >= size:
                        break
                    if board[k, j] != piece:
                        break
                    cnt += 1
                    k += 1
                k = i - 1
                while True:
                    if k < 0:
                        break
                    if board[k, j] != piece:
                        break
                    cnt += 1
                    k -= 1
                if cnt >= 5:
                    return piece

                cnt = 1
                k = 1
                while True:
                    x = i - k
                    y = j - k
                    if x < 0 or y < 0:
                        break
                    if board[x, y] != piece:
                        break
                    cnt += 1
                    k += 1
                k = 1
                while True:
                    x = i + k
                    y = j + k
                    if x >= size or y >= size:
                        break
                    if board[x, y] != piece:
                        break
                    cnt += 1
                    k += 1
                if cnt >= 5:
                    return piece

                cnt = 1
                k = 1
                while True:
                    x = i + k
                    y = j - k
                    if x >= size or y < 0:
                        break
                    if board[x, y] != piece:
                        break
                    cnt += 1
                    k += 1
                k = 1
                while True:
                    x = i - k
                    y = j + k
                    if x < 0 or y >= size:
                        break
                    if board[x, y] != piece:
                        break
                    cnt += 1
                    k += 1
                if cnt >= 5:
                    return piece
        return None

    # def find_check(self, color, depth, only_four):
    #     global MAX_SCORE, MIN_SCORE, last_min_point, last_max_point
    #     if depth is None:
    #         depth = check_depth
    #     if depth <= 0:
    #         return None, None
    #     if only_four:
    #         MAX_SCORE = MY_FLUSH_FOUR if color == self.color else RIVAL_FLUSH_FOUR
    #         MIN_SCORE = FIVE
    #         for i in range(1, depth + 1):
    #             last_max_point = None
    #             last_min_point = None
    #             result = self._max(color, i)
    #             if result is not None:
    #                 return MY_LIVE_FOUR if color == self.color else RIVAL_LIVE_FOUR, len(result)
    #         return None, None
    #     else:
    #         MIN_SCORE = MY_JUMP_LIVE_THREE if color == self.color else RIVAL_JUMP_LIVE_THREE
    #         MIN_SCORE = MY_FLUSH_FOUR if color == self.color else RIVAL_FLUSH_FOUR
    #         for i in range(1, depth + 1):
    #             last_max_point = None
    #             last_min_point = None
    #             result = self._max(color, i)
    #             if result is not None:
    #                 return MY_JUMP_LIVE_THREE * 2 if color == self.color else RIVAL_JUMP_LIVE_THREE * 2, len(result)
    #         return None, None


def cmp_score_1(a, b):
    if equal(a[SCORE_IDX], b[SCORE_IDX]):
        if a[SCORE_IDX] > 0:
            if a[STEP_IDX] != b[STEP_IDX]:
                return a[STEP_IDX] - b[STEP_IDX]
        elif a[STEP_IDX] != b[STEP_IDX]:
            return b[STEP_IDX] - a[STEP_IDX]
    return b[SCORE_IDX] - a[SCORE_IDX]


def cmp_score_2(a, b):
    if equal(a[SCORE_IDX], b[SCORE_IDX]):
        return b[MS_IDX] + b[RS_IDX] - a[MS_IDX] - a[RS_IDX]
    else:
        return b[SCORE_IDX] - a[SCORE_IDX]


def is_in_star(piece, pieces):
    if not piece or not pieces:
        return False
    for p in pieces:
        a_idx = p[LOC_IDX]
        b_idx = piece[LOC_IDX]
        row_dis = a_idx[0] - b_idx[0]
        col_dis = a_idx[1] - b_idx[1]
        if abs(row_dis) > 4 or abs(col_dis > 4):
            return False
        if row_dis != 0 and col_dis != 0 and row_dis != col_dis:
            return False
    return True


def equal(a, b):
    b = b or 0.01
    return b / math_threshold <= a <= b * math_threshold if b >= 0 \
        else b * math_threshold <= a <= b / math_threshold


def greater(a, b):
    return a >= (b + 0.1) * math_threshold if b >= 0 else a >= (b + 0.1) / math_threshold


def greater_or_equal(a, b):
    return equal(a, b) or greater(a, b)


def less(a, b):
    return a <= (b - 0.1) / math_threshold if b >= 0 else a <= (b - 0.1) * math_threshold


def less_or_equal(a, b):
    return equal(a, b) or less(a, b)
