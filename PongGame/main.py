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
from kivy.lang.builder import Builder
from kivy.uix.boxlayout import BoxLayout

class PongPaddle(Widget):
    score = NumericProperty(0)

    def bounce_ball(self, ball):
        if self.collide_widget(ball):
            vx, vy = ball.velocity
            offset = (ball.center_y - self.center_y) / (self.height / 2)
            bounced = Vector(-1 * vx, vy)
            vel = bounced * 1.1
            ball.velocity = vel.x, vel.y + offset
            return 5
        return 0
    v_y = NumericProperty(0)
    def move(self,a:NumericProperty):
        self.center_y += a
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
    qagt = Q_Agent()
    qagt2 = Q_Agent()
    oldscore = 0
    ori_pos = []
    def serve_ball(self, vel=(4, 0)):
        self.ball.center = self.center
        self.ball.velocity = vel
        # self.qagt.boundray = self.height
        self.oldscore = self.player2.score
        self.ori_pos = self.player2.pos


    def update(self, dt):
        # call ball.move and other stuff
        self.qagt.bar_pos = self.player2.pos
        self.qagt.new_b_pos = self.ball.pos
        self.qagt2.bar_pos = self.player1.pos
        self.qagt2.new_b_pos = self.ball.pos
        a = self.qagt.getAction()
        a2 = self.qagt2.getAction()
        # print("out:",a)
        # print(self.player2.pos,self.ball.pos)
        self.player2.move(a)
        # self.player1.move(a2)
        self.ball.move()
        r = 0
        r2 = 0
        # r = ((self.ball.pos[0]-self.player2.pos[0])**2+(self.ball.pos[1]-self.player2.pos[1])**2)**0.5
        # print("rewd:",r,self.ball.x, self.player2.x)
        
        
    
        r2 = self.player1.bounce_ball(self.ball)
        r = self.player2.bounce_ball(self.ball)

        
        # bounce off top and bottom
        if (self.ball.y < 0) or (self.ball.top > self.height):
            self.ball.velocity_y *= -1

        # bounce off left and right
        # if (self.ball.x < 0) or (self.ball.right > self.width):
        #     self.ball.velocity_x *= -1

        # went of to a side to score point?
        # r = 0
        if self.ball.x < self.x:
            self.player2.score += 1
            self.serve_ball(vel=(4, 0))
            r2 = -5
            self.qagt.iters -=1
            self.qagt.iters -=1
        elif self.ball.x > self.width:
            self.player1.score += 1
            r = -5
            self.serve_ball(vel=(-4, 0))
            self.qagt.iters -=1
            self.qagt.iters -=1
            # r = -5
        print("rwd: ",r)
        if self.qagt.iters > 0:
            self.qagt.q_learn(r)
        if self.qagt2.iters > 0:
            self.qagt2.q_learn(r2)
    
    def on_touch_move(self, touch):
        if touch.x < self.width/3:
            self.player1.center_y = touch.y
        # if touch.x > self.width - self.width/3:
        #     self.player2.center_y = touch.y


class PongApp(App):
    def build(self):
        Builder.load_file('pong1.kv')
        game = PongGame()
        # print("h")
        game.serve_ball()
        Clock.schedule_interval(game.update, 1.0/60.0)
        return game


if __name__ == '__main__':
    PongApp().run()