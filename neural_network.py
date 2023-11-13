import numpy as np
from keras.datasets import mnist
import pygame
import pygame.locals as pl
from PIL import Image

#--- Data preparation START ---
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
SCALE_FACTOR = 255

WIDTH = X_train.shape[1]
HEIGHT = X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], WIDTH*HEIGHT).T / SCALE_FACTOR
X_test = X_test.reshape(X_test.shape[0], WIDTH*HEIGHT).T / SCALE_FACTOR
#--- Data preparation END ---

def guess_the_params(size):
    w1 = np.random.rand(10, size) -0.5
    b1 = np.random.rand(10, 1) -0.5
    w2 = np.random.rand(10, 10) -0.5
    b2 = np.random.rand(10, 1) -0.5
    return w1, b1, w2, b2

def relu(z):
    return np.maximum(0, z)

def softmax(Z):
    exp = np.exp(Z - np.max(Z))
    return exp / exp.sum(axis=0)

def propagation(w1, b1, w2, b2, x):
    y1 = w1.dot(x) + b1
    a1 = relu(y1)
    y2 = w2.dot(a1) + b2
    a2 = softmax(y2)
    return y1, a1, y2, a2

def derivative_relu(z):
    return z > 0

def one_hot_encoding(y):
    y = y.astype(np.int64)
    one_hot = np.zeros((y.max() + 1, y.size))
    one_hot[y, np.arange(y.size)] = 1
    return one_hot

def back_propagation(X, Y, A1, A2, W2, Z1, m):
    one_hot_res = one_hot_encoding(Y)
    dZ2 = 2*(A2 - one_hot_res)
    dW2 = 1/m * (dZ2.dot(A1.T))
    db2 = 1/m * np.sum(dZ2, 1)
    dZ1 = W2.T.dot(dZ2) * derivative_relu(Z1)
    dW1 = 1/m * (dZ1.dot(X.T))
    db1 = 1/m * np.sum(dZ1, 1)

    return dW1, db1, dW2, db2

def update_params(w1, b1, w2, b2, der_w1, der_b1, der_w2, der_b2, l_rate):
    w1 = w1 - l_rate * der_w1
    b1 = b1 - l_rate * np.reshape(der_b1, (10, 1))

    w2 = w2 - l_rate * der_w2
    b2 = b2 - l_rate * np.reshape(der_b2, (10, 1))

    return w1, b1, w2, b2

def get_predictions(a2):
    return np.argmax(a2, 0)


def calculate_accuracy(predictions, label):
    return np.sum(predictions == label) / label.size


def gradient_descent(x, y, l_rate, epochs):
    size, m = x.shape
    w1, b1, w2, b2 = guess_the_params(size)

    for i in range(epochs):

        z1, a1, z2, a2 = propagation(w1, b1, w2, b2, x)
        der_w1, der_b1, der_w2, der_b2 = back_propagation(x, y, a1, a2, w2, z1, m)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, der_w1, der_b1, der_w2, der_b2, l_rate)

        if i % 10 == 0:
            print(f"Epoch: {i + 1} / {epochs}")
            prediction = get_predictions(a2)
            print(f'{calculate_accuracy(prediction, y):.15%}')

    return w1, b1, w2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 350)

def model_test(index, x, y, w1, b1, w2, b2):
    vect_X = x[:, index, None]
    _, _, _, a2 = propagation(w1, b1, w2, b2, vect_X)
    result = get_predictions(a2)
    label = y[index]
    print("Prediction: ", result)
    print("Label: ", label)

model_test(0, X_test, Y_test, W1, b1, W2, b2)
model_test(526, X_test, Y_test, W1, b1, W2, b2)

def test_by_drawing(w1, b1, w2, b2):
    # Constants
    WIDTH, HEIGHT = 280, 280
    BLACK = (255, 255, 255)
    WHITE = (0, 0, 0)
    PEN_SIZE = 5

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Number drawer")

    drawing = False
    last_pos = (0, 0)
    image = pygame.Surface((WIDTH, HEIGHT))
    image.fill(WHITE)

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pl.QUIT:
                running = False
            if event.type == pl.MOUSEBUTTONDOWN:
                drawing = True
            if event.type == pl.MOUSEBUTTONUP:
                drawing = False
            if event.type == pl.MOUSEMOTION:
                if drawing:
                    mouse_x, mouse_y = event.pos
                    pygame.draw.line(image, BLACK, last_pos, (mouse_x, mouse_y), PEN_SIZE)
                last_pos = event.pos
        if drawing:
            pygame.display.update()

        screen.fill(WHITE)
        screen.blit(image, (0, 0))
        pygame.display.update()

    pygame.image.save(image, "number.png")
    img = Image.open("number.png")

    # A different way to normalize the data before inserting them to the network
    img = img.resize((28, 28))
    img = img.convert('L')  # Convert to grayscale
    img = np.asarray(img)
    img = img.reshape(784, 1) / 255.0

    _, _, _, a2 = propagation(w1, b1, w2, b2, img)
    result = get_predictions(a2)
    print("The number you drawn is: ", result)
    pygame.quit()

while True:
    test_by_drawing(W1, b1, W2, b2)
