# Skeleton File for CS 251 Project 3
# You can use this file or your file from Project 1
# Delete unnecessary code before handing in

import tkinter as tk
import math
import random
import numpy as np

def init_axes():
    return np.matrix[[0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1, 1]]

# create a class to build and manage the display
class DisplayApp:

    def __init__(self, width, height):

        # create a tk object, which is the root window
        self.root = tk.Tk()

        # width and height of the window
        self.initDx = width
        self.initDy = height

        # axes 
        self.axes = init_axes()
        self.buildAxes()

        # set up the geometry for the window
        self.root.geometry( "%dx%d+50+30" % (self.initDx, self.initDy) )

        # set the title of the window
        self.root.title("Viewing Axes")

        # set the maximum size of the window for resizing
        self.root.maxsize( 1024, 768 )

        # bring the window to the front
        self.root.lift()

        # setup the menus
        self.buildMenus()

        # build the controls
        self.buildControls()

        # build the objects on the Canvas
        self.buildCanvas()

        # set up the key bindings
        self.setBindings()

        # Create a View object and set up the default parameters

        # Create the axes fields and build the axes

        # set up the application state
        self.objects = []
        self.data = None

    def buildMenus(self):
        
        # create a new menu
        self.menu = tk.Menu(self.root)

        # set the root menu to our new menu
        self.root.config(menu = self.menu)

        # create a variable to hold the individual menus
        self.menulist = []

        # create a file menu
        filemenu = tk.Menu( self.menu )
        self.menu.add_cascade( label = "File", menu = filemenu )
        self.menulist.append(filemenu)


        # menu text for the elements
        menutext = [ [ 'Open...  \xE2\x8C\x98-O', '-', 'Quit  \xE2\x8C\x98-Q' ] ]

        # menu callback functions
        menucmd = [ [self.handleOpen, None, self.handleQuit]  ]
        
        # build the menu elements and callbacks
        for i in range( len( self.menulist ) ):
            for j in range( len( menutext[i]) ):
                if menutext[i][j] != '-':
                    self.menulist[i].add_command( label = menutext[i][j], command=menucmd[i][j] )
                else:
                    self.menulist[i].add_separator()

    # create the canvas object
    def buildCanvas(self):
        self.canvas = tk.Canvas( self.root, width=self.initDx, height=self.initDy )
        self.canvas.pack( expand=tk.YES, fill=tk.BOTH )
        return

    # build a frame and put controls in it
    def buildControls(self):

        # make a control frame
        self.cntlframe = tk.Frame(self.root)
        self.cntlframe.pack(side=tk.RIGHT, padx=2, pady=2, fill=tk.Y)

        sep = tk.Frame( self.root, height=self.initDy, width=2, bd=1, relief=tk.SUNKEN )
        sep.pack( side=tk.RIGHT, padx = 2, pady = 2, fill=tk.Y)

        # make a cmd 1 button in the frame
        self.buttons = []
        self.buttons.append( ( 'reset', tk.Button( self.cntlframe, text="Reset", command=self.handleResetButton, width=5 ) ) )
        self.buttons[-1][1].pack(side=tk.TOP)  # default side is top

        return

    # create the axis line objects in their default location
    def buildAxes(self):
        pass

    # modify the endpoints of the axes to their new location
    def updateAxes(self):
        pass

    def setBindings(self):
        self.root.bind( '<Button-1>', self.handleButton1 )
        self.root.bind( '<Button-2>', self.handleButton2 )
        self.root.bind( '<Button-3>', self.handleButton3 )
        self.root.bind( '<B1-Motion>', self.handleButton1Motion )
        self.root.bind( '<B2-Motion>', self.handleButton2Motion )
        self.root.bind( '<B3-Motion>', self.handleButton3Motion )
        self.root.bind( '<Control-q>', self.handleQuit )
        self.root.bind( '<Control-o>', self.handleModO )
        self.canvas.bind( '<Configure>', self.handleResize )
        return

    def handleResize(self, event=None):
        # You can handle resize events here
        pass


    def handleOpen(self):
        print('handleOpen')

    def handleModO(self, event):
        self.handleOpen()

    def handleQuit(self, event=None):
        print('Terminating')
        self.root.destroy()

    def handleResetButton(self):
        print('handling reset button')

    def handleButton1(self, event):
        print('handle button 1: %d %d' % (event.x, event.y))

    # rotation
    def handleButton2(self, event):
        print('handle button 2: %d %d' % (event.x, event.y))

    # scaling
    def handleButton3(self, event):
        print('handle button 3: %d %d' % (event.x, event.y))

    # translation
    def handleButton1Motion(self, event):
        print('handle button 1 motion: %d %d' % (event.x, event.y) )
    
    def handleButton2Motion(self, event):
        print('handle button 2 motion: %d %d' % (event.x, event.y) )

    def handleButton3Motion( self, event):
        print('handle button 3 motion: %d %d' % (event.x, event.y) )

    def main(self):
        print('Entering main loop')
        self.root.mainloop()

if __name__ == "__main__":
    dapp = DisplayApp(700, 500)
    dapp.main()


