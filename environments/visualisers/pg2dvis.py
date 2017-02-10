import pygame
import numpy as np

from environments.d2.line import Line

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000

TS_SCALE = 50
TS_MOVE = np.asarray((SCREEN_WIDTH/2,SCREEN_HEIGHT/2))

def _ts(x):
    return int(x * TS_SCALE);

def _tv(v):
    return ( np.array((v[0],-v[1])) * TS_SCALE + TS_MOVE ).astype(int)

def _tc(c):
    return np.clip( np.asarray( (c[0] * 255, c[1] * 255, c[2] * 255, c[3] * 255) ), 0, 255 )

class PyGame2D:
    def __init__(self, width=600, height=400, bot_radius=0.3):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
        self.bot_radius = bot_radius
        pass

    def draw_env(self, env):
        for shape in env.shapelist:
            from environments.d2.circle import Circle

            if type(shape) is Circle:
                pygame.draw.circle(self.screen, shape.color, shape.pos, shape.radius)

        # Also draw the agent
        pygame.draw.circle(self.screen, (0, 0, 0), env.position, 0.3)
        pygame.draw.line(self.screen, (0, 0, 0), env.position, env.position + 1.0 * env.f_direction, width=5 )

    def update(self):
        pygame.display.update()

    def draw_path_cmp(self, env, path, path_pred, attns, rewards):

        clock = pygame.time.Clock()

        for p,pred,attn,rews in zip(path,path_pred,attns,rewards):
            # Two frames per second.
            clock.tick(1)

            self.screen.fill((255,255,255))
            done = False
            for event in pygame.event.get():  # User did something
                if event.type == pygame.QUIT:
                    done = True

            if done:
                break

            for shape in env.shape_list:
                from environments.d2.circle import Circle

                if isinstance(shape, Circle):
                    pygame.draw.circle(self.screen, _tc(shape.color), _tv(shape.pos), _ts(shape.radius))
                elif isinstance(shape, Line):
                    pygame.draw.line(self.screen, _tc(shape.color), _tv(shape.start_pos), _tv(shape.end_pos), 3)
            # Also draw the agent
            pygame.draw.circle(self.screen, (0, 0, 0), _tv(p[2]), _ts(0.2))
            pygame.draw.line(self.screen, (0, 0, 0), _tv(p[2]), _tv(p[2] + 0.4 * p[3]), 3 )

            # Draw a string of boxes to illustrate the visualisation.
            W = p[0][0].shape[0]
            visYt2 = 0.8 * SCREEN_HEIGHT
            visYt = 0.8 * SCREEN_HEIGHT
            visYb = 1.0 * SCREEN_HEIGHT
            visX = np.linspace(0, SCREEN_WIDTH, W + 1)

            for impix in zip( p[0][0], visX, visX[1:], pred, attn ):
                pygame.draw.rect(self.screen, _tc(impix[0]), (impix[1], visYt, SCREEN_WIDTH / W, SCREEN_WIDTH / W))
                pygame.draw.rect(self.screen, (255, 255, 255), (impix[1], visYt, SCREEN_WIDTH / W, SCREEN_WIDTH / W),
                                 1)
                pygame.draw.rect(self.screen, _tc(impix[3]), (impix[1], visYt + 0.01 * SCREEN_HEIGHT, SCREEN_WIDTH / W, SCREEN_WIDTH / W))
                pygame.draw.rect(self.screen, (255, 255, 255),
                                 (impix[1], visYt + 0.01 * SCREEN_HEIGHT, SCREEN_WIDTH / W, SCREEN_WIDTH / W),
                                 1)
                visYtemp = visYt + 0.01 * SCREEN_HEIGHT

                """for atmod in impix[4]:
                    visYtemp = visYtemp + 0.01 * SCREEN_HEIGHT
                    pygame.draw.rect(self.screen, _tc((atmod,atmod,atmod,1.0)),
                                 (impix[1], visYtemp, SCREEN_WIDTH / W, SCREEN_WIDTH / W))
                    pygame.draw.rect(self.screen, (255, 255, 255),
                                 (impix[1], visYtemp, SCREEN_WIDTH / W, SCREEN_WIDTH / W),
                                 1)
                """
            pygame.draw.rect(self.screen, _tc((rews[0],rews[0],rews[0],1.0)), ( 0,visYt + 0.02 * SCREEN_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH / W))

            pygame.display.update()

    def draw_path(self, env, path):
        clock = pygame.time.Clock()

        for p in path:
            # Two frames per second.
            clock.tick(10)

            self.screen.fill((255,255,255))
            done = False
            for event in pygame.event.get():  # User did something
                if event.type == pygame.QUIT:
                    done = True

            if done:
                break

            for shape in env.shape_list:
                from environments.d2.circle import Circle

                if isinstance(shape, Circle):
                    pygame.draw.circle(self.screen, _tc(shape.color), _tv(shape.pos), _ts(shape.radius))
                elif isinstance(shape, Line):
                    pygame.draw.line(self.screen, _tc(shape.color), _tv(shape.start_pos), _tv(shape.end_pos), 3)

            # Also draw the agent
            pygame.draw.circle(self.screen, (0, 0, 0), _tv(p[2]), _ts(0.2))
            pygame.draw.line(self.screen, (0, 0, 0), _tv(p[2]), _tv(p[2] + 0.4 * p[3]), 3)

            # Draw a string of boxes to illustrate the visualisation.
            W = p[0][0].shape[0]
            visYt = 0.8 * SCREEN_HEIGHT
            visYb = 1.0 * SCREEN_HEIGHT
            visX = np.linspace(0, SCREEN_WIDTH, W + 1)

            for impix in zip( p[0][0], visX, visX[1:] ):
                pygame.draw.rect(self.screen, _tc(impix[0]), (impix[1], visYt, SCREEN_WIDTH / W, 0.03 * SCREEN_HEIGHT))
                pygame.draw.rect(self.screen, (255, 255, 255), (impix[1], visYt, SCREEN_WIDTH / W, 0.03 * SCREEN_HEIGHT),
                                 1)

            pygame.display.update()