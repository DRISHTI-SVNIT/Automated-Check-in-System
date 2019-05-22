import cv2
from tkinter import *
from PIL import Image, ImageTk
from connection_to_db import Connect

refPt = []
cropping = False
f_name=[]
l_name=[]
ad_no=[]

def show_entry_fields():
        print("First Name: %s\nLast Name: %s" % (e1.get(), e2.get()))

def store_entries():
    global f_name,l_name,ad_no
    f_name.append(e1.get())
    l_name.append(e2.get())
    ad_no.append(e3.get())
        

def display():
    global f_name,l_name,ad_no
    for i in range(len(f_name)):
        print(f_name[i],l_name[i],ad_no[i])
 
def click_and_crop(event, x, y, flags, param):
    global refPt, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
 
	
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False
 
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

            
cap = cv2.VideoCapture(0)

if (cap.isOpened()== False): 
    print("Error opening video stream or file")

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    key = cv2.waitKey(1) & 0xFF
    frame=cv2.flip(frame,1)
    if ret == True:
        

        cv2.imshow('frame',frame)
        # Press Q on keyboard to  exit
        if key == ord('q'):
            break
        
        if key == ord('o') :
            image=frame.copy()
            clone=image.copy()
            cv2.namedWindow("image")
            cv2.setMouseCallback("image", click_and_crop)
            
            '''
            image=frame.copy()
            clone=frame.copy() 
            '''
            while True:
                cv2.imshow("image", image)
                key1 = cv2.waitKey(1) & 0xFF
                if key1 == ord("r"):
                    image = clone.copy()
                    
                elif key1 == ord("c"):
                    break
            
            if len(refPt) == 2:
                roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
                roi=cv2.resize(roi,(50,50))
                cv2.imshow("ROI", roi)
                cv2.imwrite('croped_temp.png',roi)
                #cv2.waitKey(0)                
    else: 
        break
        
cap.release()
cv2.destroyAllWindows()

master = Tk()

load = Image.open('croped_temp.png')
render= ImageTk.PhotoImage(load)
Label(master,image=render).grid(row=10)
Label(master, text="First Name").grid(row=0,column=0)
Label(master, text="Last Name").grid(row=1,column=0)
Label(master, text="Ad no.").grid(row=2,column=0)

e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)

Button(master, text='Quit', command=master.quit).grid(row=3, column=0, sticky=W, pady=4)
Button(master, text='Store', command=store_entries).grid(row=3, column=1, sticky=W, pady=4)
Button(master,text='Display', command=display).grid(row=3,column=2,sticky=W, pady=4)

master.mainloop()

c=Connect()
c.push(f_name[0],l_name[0],ad_no[0])