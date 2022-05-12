# from typing import overload
from kivy.app import App
from kivy.uix.label import Label
from  kivy.uix.gridlayout import GridLayout
from kivy.lang.builder import Builder
class ChartView(GridLayout):
    
    def getQuery(self):
        return 1

class MyApp(App):

    def build(self):
        mview = ChartView()
        return mview


if __name__ == '__main__':
    MyApp().run()
