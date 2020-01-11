# Qingbo Liu 
#
#
# CS 251
# Spring 2020

import tkinter as tk
import math
import random

ListBoxSelections = ["Random", "Gaussian"]

# Modal dialog template 
class Dialog(tk.Toplevel):
    def __init__(self, parent, title = None):

        tk.Toplevel.__init__(self, parent)
        self.transient(parent)

        if title:
            self.title(title)

        self.parent = parent

        self.result = None

        body = tk.Frame(self)
        self.initial_focus = self.body(body)
        body.pack(padx=5, pady=5)

        self.buttonbox()

        self.grab_set()

        if not self.initial_focus:
            self.initial_focus = self

        self.protocol("WM_DELETE_WINDOW", self.cancel)

        self.geometry("+%d+%d" % (parent.winfo_rootx()+50,
                                  parent.winfo_rooty()+50))

        self.initial_focus.focus_set()

        self.wait_window(self)

    #
    # construction hooks

    def body(self, master):
        # create dialog body.  return widget that should have
        # initial focus.  this method should be overridden

        pass

    def buttonbox(self):
        # add standard button box. override if you don't want the
        # standard buttons

        box = tk.Frame(self)

        w = tk.Button(box, text="OK", width=10, command=self.ok, default=tk.ACTIVE)
        w.pack(side=tk.LEFT, padx=5, pady=5)
        w = tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

        box.pack()

    #
    # standard button semantics

    def ok(self, event=None):

        if not self.validate():
            self.initial_focus.focus_set() # put focus back
            return

        self.withdraw()
        self.update_idletasks()

        self.apply()

        self.cancel()

    def cancel(self, event=None):
        # put focus back to the parent window
        self.parent.focus_set()
        self.destroy()

    #
    # command hooks

    def validate(self):

        return 1 # override

    def apply(self):

        pass # override

class RandomPointDialog(Dialog): 
    def __init__(self, parent, lower, upper, title = None):
        self.lower = lower 
        self.upper = upper 
        self.cancelled = True 
        self.numPoints = min 
        super().__init__(parent, title)

    def body(self, master):
        v = tk.StringVar()
        v.set("10")
        widget = tk.Entry(master, textvariable=v)
        widget.pack()

        widget.focus_set()
        widget.select_range(0, tk.END)
	
        self.text = v 

    def validate(self):
        s = self.text.get()
        try:
            v = int(s) 
        except TypeError:
            return False 

        if v < self.lower or v > self.upper:
            return False 
        else:
            self.numPoints = v 
            return True 

    def apply(self):
        self.cancelled = False 

    def userCancelled(self):
        return self.cancelled

    def getNumPoints(self):
        return self.numPoints


# create a class to build and manage the display
class DisplayApp:

    def __init__(self, width, height):

        # create a tk object, which is the root window
        self.root = tk.Tk()

        # width and height of the window
        self.initDx = width
        self.initDy = height

        # set up the geometry for the window
        self.root.geometry( "%dx%d+50+30" % (self.initDx, self.initDy) )

        # set the title of the window
        self.root.title("Please give me a better name")

        # set the maximum size of the window for resizing
        self.root.maxsize( 1600, 900 )

        # setup the menus
        self.buildMenus()

        # build the controls
        self.buildControls()

        # build the Canvas
        self.buildCanvas()

        # bring the window to the front
        self.root.lift()

        # - do idle events here to get actual canvas size
        self.root.update_idletasks()

        # now we can ask the size of the canvas
        print(self.canvas.winfo_geometry())

        # set up the key bindings
        self.setBindings()

        # set up the application state
        self.objects = [] # list of data objects that will be drawn in the canvas
        self.data = None # will hold the raw data someday.
        self.baseClick = None # used to keep track of mouse movement

    def buildMenus(self):
        
        # create a new menu
        menu = tk.Menu(self.root)

        # set the root menu to our new menu
        self.root.config(menu = menu)

        # create a variable to hold the individual menus
        menulist = []

        # create a file menu
        filemenu = tk.Menu( menu )
        menu.add_cascade( label = "File", menu = filemenu )
        menulist.append(filemenu)

        # create another menu for kicks
        cmdmenu = tk.Menu( menu )
        menu.add_cascade( label = "Command", menu = cmdmenu )
        menulist.append(cmdmenu)

        # menu text for the elements
        # the first sublist is the set of items for the file menu
        # the second sublist is the set of items for the option menu
        menutext = [ [ '-', '-', 'Quit  Ctl-Q', 'Clear Ctl-N' ],
                     [ 'Command 1', '-', '-' ] ]

        # menu callback functions (note that some are left blank,
        # so that you can add functions there if you want).
        # the first sublist is the set of callback functions for the file menu
        # the second sublist is the set of callback functions for the option menu
        menucmd = [ [None, None, self.handleQuit, self.clearData],
                    [self.handleMenuCmd1, None, None] ]
        
        # build the menu elements and callbacks
        for i in range( len( menulist ) ):
            for j in range( len( menutext[i]) ):
                if menutext[i][j] != '-':
                    menulist[i].add_command( label = menutext[i][j], command=menucmd[i][j] )
                else:
                    menulist[i].add_separator()

    # create the canvas object
    def buildCanvas(self):
        self.canvas = tk.Canvas( self.root, width=self.initDx, height=self.initDy )
        self.canvas.pack( expand=tk.YES, fill=tk.BOTH )
        return

    # build a frame and put controls in it
    def buildControls(self):

        ### Control ###
        # make a control frame on the right
        rightcntlframe = tk.Frame(self.root)
        rightcntlframe.pack(side=tk.RIGHT, padx=2, pady=2, fill=tk.Y)

        # make a separator frame
        sep = tk.Frame( self.root, height=self.initDy, width=2, bd=1, relief=tk.SUNKEN )
        sep.pack( side=tk.RIGHT, padx = 2, pady = 2, fill=tk.Y)

        # use a label to set the size of the right panel
        label = tk.Label( rightcntlframe, text="Control Panel", width=20 )
        label.pack( side=tk.TOP, pady=10 )

        # make a menubutton
        self.colorOption = tk.StringVar( self.root )
        self.colorOption.set("black")
        colorMenu = tk.OptionMenu( rightcntlframe, self.colorOption, 
                                        "black", "blue", "red", "green" ) # can add a command to the menu
        colorMenu.pack(side=tk.TOP)

        # "Update Color" button 
        button = tk.Button( rightcntlframe, text="Update Color", 
                               command=self.updatePointsColor )
        button.pack(side=tk.TOP)  # default side is top

	# "Random Points" button 
        button = tk.Button( rightcntlframe, text="Random Points", 
                               command=self.createRandomPoints )
        button.pack(side=tk.TOP)

	# Listbox 
        listbox = tk.Listbox(rightcntlframe, selectmode=tk.SINGLE)
        for item in ListBoxSelections:
            listbox.insert(tk.END, item)
        listbox.selection_set("0")
        listbox.pack(side=tk.TOP)
        self.listbox = listbox

        return

    def setBindings(self):
        # bind mouse motions to the canvas
        self.canvas.bind( '<Button-1>', self.handleMouseButton1 )
        self.canvas.bind( '<Control-Button-1>', self.handleMouseButton2 )
        self.canvas.bind( '<Shift-Button-1>', self.handleMouseButton3 )
        self.canvas.bind( '<B1-Motion>', self.handleMouseButton1Motion )
        self.canvas.bind( '<Control-B1-Motion>', self.handleMouseButton2Motion )
        self.canvas.bind( '<Shift-B1-Motion>', self.handleMouseButton3Motion )

        # bind command sequences to the root window
        self.root.bind( '<Control-q>', self.handleQuit )
        self.root.bind( '<Control-n>', self.clearData )

    def handleQuit(self, event=None):
        print( 'Terminating')
        self.root.destroy()

    def clearData(self, event=None): 
        for obj in self.objects:
            self.canvas.delete(obj)
        self.objects = [] 

    def updatePointsColor(self):
        for obj in self.objects:
            self.canvas.itemconfig(obj, fill=self.colorOption.get() )


    def handleMenuCmd1(self):
        print( 'handling menu command 1')

    def handleMouseButton1(self, event):
        print( 'handle mouse button 1: %d %d' % (event.x, event.y))
        self.baseClick = (event.x, event.y)

    def handleMouseButton2(self, event):
        self.baseClick = (event.x, event.y)

        dx = 3
        rgb = "#%02x%02x%02x" % (random.randint(0, 255), 
                                 random.randint(0, 255), 
                                 random.randint(0, 255) )
        oval = self.canvas.create_oval( event.x - dx,
                                        event.y - dx, 
                                        event.x + dx, 
                                        event.y + dx,
                                        fill = rgb,
                                        outline='')
        self.objects.append( oval )

        print( 'handle mouse button 2: %d %d' % (event.x, event.y))

    def handleMouseButton3(self, event):
        self.baseClick = (event.x, event.y)
        print( 'handle mouse button 3: %d %d' % (event.x, event.y))

    # This is called if the first mouse button is being moved
    def handleMouseButton1Motion(self, event):
        # calculate the difference
        diff = ( event.x - self.baseClick[0], event.y - self.baseClick[1] )

        # update objects location 
        for obj in self.objects:
            loc = self.canvas.coords(obj) 
            self.canvas.coords(obj, 
                                loc[0] + diff[0],
                                loc[1] + diff[1], 
                                loc[2] + diff[0], 
                                loc[3] + diff[1] )

        # update base click
        self.baseClick = ( event.x, event.y )
        print( 'handle button1 motion %d %d' % (diff[0], diff[1]))

            
    # This is called if the second button of a real mouse has been pressed
    # and the mouse is moving. Or if the control key is held down while
    # a person moves their finger on the track pad.
    def handleMouseButton2Motion(self, event):
        # diff = ( event.x - self.baseClick[0], event.y - self.baseClick[1] )
        # diff_area = abs(diff[0]) * abs(diff[1])
        # width = self.canvas.winfo_width()
        # height = self.canvas.winfo_height()
        # dx = (diff_area / (width * height)) * 10

        # for obj in self.objects:
            # loc = self.canvas.coords(obj)
            # self.canvas.coords(obj, 
                                # loc[0] - dx,
                                # loc[1] 

        print( 'handle button 2 motion %d %d' % (event.x, event.y) )

    def handleMouseButton3Motion(self, event):
        print( 'handle button 3 motion %d %d' % (event.x, event.y) )
        
    def createRandomPoints(self, event = None):
        # check what generation mode is selected 
        item = list(map(int, self.listbox.curselection()))
        if not item: 
            print("no selection in listbox")
            return 
        else:
            selection = ListBoxSelections[item[0]]

        # ask user for number of random points 
        dialog = RandomPointDialog(self.root, 1, 30, "Number of Random Points")
        if dialog.userCancelled():
            return 
        N = dialog.getNumPoints()


        # create random points 
        dx = 3 
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if selection == "Random":
            for _ in range(N): 
                x = random.randint(0, width)
                y = random.randint(0, height)
                pt = self.canvas.create_oval( x-dx, y-dx, x+dx, y+dx,
                                            fill=self.colorOption.get(), outline='' )
                self.objects.append(pt)
        elif selection == "Gaussian":
            width_mean = width/2
            width_std = width/6
            height_mean = height/2
            height_std = height/6
            for _ in range(N): 
                x = random.gauss(width_mean, width_std)
                y = random.gauss(height_mean, height_std)
                pt = self.canvas.create_oval( x-dx, y-dx, x+dx, y+dx,
                                            fill=self.colorOption.get(), outline='' )
                self.objects.append(pt)

    def main(self):
        print( 'Entering main loop')
        self.root.mainloop()



if __name__ == "__main__":
    dapp = DisplayApp(1200, 675)
    dapp.main()


