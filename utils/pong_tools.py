import numpy as np

def get_all_coords(I):
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1e
    opponent_slice = I[:, 9:10]
    opponent_x = 10.
    opponent_y = ((np.argmax(opponent_slice, axis=0) + (
        opponent_slice.shape[0] - np.argmax(np.array(list(reversed(opponent_slice))), axis=0))) / 2)[0]
    self_slice = I[:, 70:71]
    self_x = 70.
    self_y = ((np.argmax(self_slice, axis=0) + (
        self_slice.shape[0] - np.argmax(np.array(list(reversed(self_slice))), axis=0))) / 2)[0]
    ball_slice = I[:, 10:70]
    ball_x = 10. + np.argmax(np.sum(ball_slice, axis=0))
    ball_y = np.argmax(np.sum(ball_slice, axis=1))
    return float(opponent_y - self_y), float(ball_x - self_x), float(ball_y - self_y)
    # return self_x, self_y, opponent_x, opponent_y, ball_x, ball_y


# downsampling
def prepro(I, I_prev):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    opp_rel_y, ball_rel_x, ball_rel_y = get_all_coords(I)
    opp_rel_y_prev, ball_rel_x_prev, ball_rel_y_prev = get_all_coords(I_prev)
    return np.array([opp_rel_y, ball_rel_x, ball_rel_y, opp_rel_y_prev, ball_rel_x_prev, ball_rel_y_prev])

def maxPixelValue(I, I_prev):
    for k in range(np.shape(I)[2]):
        for i in range(np.shape(I)[0]):
            for j in range(np.shape(I)[1]):
                I[i, j, k] = max(I[i, j, k], I_prev[i, j, k])
    return I

def rescale(I):
    I = I[35:195]  # crop
    I = I[::2, ::2]  # downsample by factor of 2
    return I

def rgb2gray(I):
    gray = 0.299*I[:, :, 0]
    gray += 0.587 * I[:, :, 1]
    gray += 0.114 * I[:, :, 2]

    return gray