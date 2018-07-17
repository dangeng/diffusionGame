import numpy as np
import pygame
from scipy.ndimage import convolve

HEIGHT = 100
WIDTH = 100
DISPLAY_HEIGHT = 500
DISPLAY_WIDTH = 500
quota = 100

edgeKernel = np.array([[1, 1, 1],
                       [-1, 0, 1],
                       [-1, -1, -1]])

diffKernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]]) / 4.0
diffKernel = np.array([[1, 4, 6, 4, 1],
                       [4, 16, 24, 16, 4],
                       [6, 24, -220, 24, 6],
                       [4, 16, 24, 16, 4],
                       [1, 4, 6, 4, 1]]) / 220.0

diffKernel = np.stack((np.zeros_like(diffKernel), diffKernel, np.zeros_like(diffKernel)), axis=2)



def diffuse(densities):
    #return densities + convolve2d(densities, diffKernel*1, boundary='wrap', mode='same')
    return densities + convolve(densities, diffKernel*.5)

def react(densities):
    num_occupants = np.sum((densities.astype(int) > 0).astype(int), axis=2)

    argmax = np.argmax(densities, axis=2)

    max_densities = np.max(densities, axis=2)
    max_densities = np.repeat(max_densities[:,:,np.newaxis], 3, axis=2)
    shifted = densities - max_densities

    zeros = shifted == 0
    shifted[zeros] = 1e10   # large number

    new_max_densities = np.min(abs(shifted), axis=2)

    new_densities = np.zeros_like(densities)
    for i in range(densities.shape[0]):
        for j in range(densities.shape[1]):
            if num_occupants[i,j] > 1:
                new_densities[i,j,argmax[i,j]] = new_max_densities[i,j]
            else:
                new_densities[i,j] = densities[i,j]

    return new_densities

def add_sources(densities, sources):
    sources = sources.copy()
    if sources[:,:,0].sum() == 0:
        sources[10, 50, 0] = 1
    if sources[:,:,1].sum() == 0:
        sources[90, 50, 1] = 1

    mask = densities.astype(int) > 0
    sources *= mask

    # Normalize while avoiding nans
    sources[:,:,:2] = sources[:,:,:2] / sources[:,:,:2].sum(axis=(0,1))

    probs = sources.flatten() / float(sources.flatten().sum())
    for i in range(5):
        idx = np.random.choice(sources.flatten().shape[0], p=probs)
        idx = np.unravel_index(idx, densities.shape)
        densities[idx] = 255
    #return densities + sources
    return densities

def limit_densities(densities, ceiling):
    mask = densities > ceiling
    densities[mask] = ceiling
    return densities

def calculate_scores(densities):
    return np.sum(densities.astype(int) > 0, axis=(0,1)) / float(HEIGHT) / float(WIDTH)

edges = 0
def create_image(densities):
    global edges
    boundaries  = np.zeros(densities.shape[0:2])
    for player in range(densities.shape[2]):
        d = densities[:,:,player].astype(int)
        d = (d == 0).astype(int)
        edges = abs(convolve(d, edgeKernel)) * 255
        boundaries = np.maximum(edges, boundaries)

    boundaries = np.stack((boundaries, boundaries, boundaries), axis=2)
    return np.maximum(densities, boundaries)

gen = lambda x: np.random.uniform(0,1,(100,100))

def run(genSourceR, genSourceG, visuals=False):
    pygame.init()

    if visuals:
        display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
        image = np.zeros((WIDTH,HEIGHT,3))
        surf = pygame.surfarray.make_surface(image)

    densities = np.zeros((WIDTH,HEIGHT,3))
    densities[10,50,0] = 255

    densities[90,50,1] = 255

    sources = np.zeros((WIDTH,HEIGHT,3))
    sources = np.random.uniform(0, 1, sources.shape)

    scores = np.zeros(3)

    # ~13 seconds for no visuals, 1000 frames
    # ~30 seconds for visuals, 1000 frames
    for _ in range(1000):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        densities[10,50,0] = 255

        densities[90,50,1] = 255

        densities = add_sources(densities, sources)
        densities = diffuse(densities)
        densities = react(densities)
        densities = limit_densities(densities, 255)
        scores += calculate_scores(densities)

        # Get new sources
        inputDensities = densities[:,:,:2]
        rSource = genSourceR(inputDensities)
        gSource = genSourceG(np.flip(inputDensities, axis=2))
        sources = np.stack((rSource, gSource, np.zeros((100,100))), axis=2)

        if visuals:
            image = create_image(densities)
            surf = pygame.surfarray.make_surface(image)
            surf = pygame.transform.scale(surf, (DISPLAY_WIDTH,DISPLAY_HEIGHT))
            display.blit(surf, (0, 0))
            pygame.display.update()

    pygame.quit()
    return scores
