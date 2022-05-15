from typing import overload
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import (
    NumericProperty, ReferenceListProperty, ObjectProperty
)
from kivy.vector import Vector
from kivy.clock import Clock
from random import randint
from q_learn import *
from blog_mtd import *
from kivy.lang.builder import Builder
from kivy.uix.boxlayout import BoxLayout
from random import *

class PongPaddle(Widget):
    score = NumericProperty(0)

    def bounce_ball(self, ball):
        if self.collide_widget(ball):
            vx, vy = ball.velocity
            offset = (ball.center_y - self.center_y) / (self.height / 2)
            bounced = Vector(-1 * vx, vy)
            vel = bounced * 1.0
            ball.velocity = vel.x, vel.y + offset
            return 5
        return 0
    def move(self,dt:NumericProperty):
        self.center_y += dt
class Marvel(BoxLayout):

    def hulk_smash(self):
        self.ids.hulk.text = "hulk: puny god!"
        self.ids["loki"].text = "loki: >_<!!!"  # alternative syntax


class PongBall(Widget):

    # velocity of the ball on x and y axis
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)

    # referencelist property so we can use ball.velocity as
    # a shorthand, just like e.g. w.pos for w.x and w.y
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    # ``move`` function will move the ball one step. This
    #  will be called in equal intervals to animate the ball
    def move(self):
        self.pos = Vector(*self.velocity) + self.pos



class PongGame(Widget):
    ball = ObjectProperty(None)
    player1 = ObjectProperty(None)
    player2 = ObjectProperty(None)
    
    oldscore = 0
    ori_pos = []
    rwad =0

    Q1 = agent(False,'right')
    Q2 = agent(True,'left')
    def serve_ball(self, vel=(4*20, 0)):
        self.ball.center = self.center
        self.ball.velocity = vel
        # self.qagt.boundray = self.height
        self.oldscore = self.player2.score
        self.ori_pos = self.player2.pos


    def update(self, dt):
        # print(self.player2.pos)
        # print(self.rwad, self.Q.tr_every)
        # if self.rwad and self.Q.tr_every == 4:
        #     self.Q.train(self.rwad)
        #     self.rwad = 0
        #     self.Q.tr_every = 0
        # else:
        #     if self.rwad:
        #         self.Q.tr_every += 1
        #         self.Q.label.extend([0 if self.rwad<0 else 1 for _ in range(len(self.Q.actions)-self.Q.tmp)])
        #         self.Q.tmp = len(self.Q.actions)
        #         self.rwad = 0
        #     action = 0
        #     if self.ball.velocity[0] > 0:

        #     # print(self.ball.pos)
        #         ballPos = [int((itm-1)/10) for itm in self.ball.pos]
        #         action = self.Q.getObvs(int(self.player1.pos[1]/10),int(self.player2.pos[1]/10),ballPos)
        ballPos = [int((itm-1)/10) for itm in self.ball.pos]
        action2 = self.Q1.train([int(self.player1.pos[1]/10),int(self.player2.pos[1]/10),ballPos],float(self.rwad),self.rwad!=0,0)
        action1 = self.Q2.train([int(self.player1.pos[1]/10),int(self.player2.pos[1]/10),ballPos],float(-self.rwad),self.rwad!=0,0)
        if self.rwad:
            self.rwad = 0
            self.player2.pos = [1575.,500.]
            self.player1.pos = [0.,500.]
        # else:
        self.ball.move()
        # print(self.ball.pos)

        # bounce of paddles
        self.player1.bounce_ball(self.ball)
        self.player2.bounce_ball(self.ball)
        self.player2.move(action2 if 0 <= self.player2.pos[1] <= 1000 else 0.)
        self.player1.move(action1 if 0 <= self.player1.pos[1] <= 1000 else 0.)

        # bounce ball off bottom or top
        if (self.ball.y < self.y) or (self.ball.top > self.top):
            self.ball.velocity_y *= -1

        # went of to a side to score point?
        if self.ball.x < self.x:
            self.player2.score += 1
            self.rwad = 1
            self.serve_ball(vel=(4*10, randint(-8,8)))
        if self.ball.right > self.width:
            self.player1.score += 1
            self.rwad = -1
            # self.Q.train()
            self.serve_ball(vel=(-4*10, randint(-8,8)))
       
    # def on_touch_move(self, touch):
    #     if touch.x < self.width/3:
    #         self.player1.center_y = touch.y
    #     if touch.x > self.width - self.width/3:
    #         self.player2.center_y = touch.y


class PongApp(App):
    def build(self):
        Builder.load_file('pong1.kv')
        game = PongGame()
        game.serve_ball()
        Clock.schedule_interval(game.update, 1.0/60.0)
        return game


if __name__ == '__main__':
        PongApp().run()
    