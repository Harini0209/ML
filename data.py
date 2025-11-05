from tkinter import*; 
from sklearn.feature_extraction.text import CountVectorizer; 
from sklearn import svm; import pandas as pd
d=pd.read_csv("one.csv"); cv=CountVectorizer(); 
X=cv.fit_transform(d["EmailText"]); 
m=svm.SVC().fit(X,d["Label"])
def check(): res=m.predict(cv.transform([e.get()]))[0];v.set(f"Result: text is {res}")
w=Tk(); w.title("Email Spam Detector"); w.configure(bg="cyan")
Label(w,text="Email Spam Detector",bg="gray",fg="white",font=("Calibri",20,"bold")).pack()
Label(w,text="Enter your Text:",bg="cyan",font=("Verdana",12)).place(x=10,y=80)
e=Entry(w,width=30); e.place(x=170,y=85); v=StringVar(); Label(w,textvariable=v,bg="cyan",font=("Verdana",12)).place(x=10,y=150)
Button(w,text="Submit",bg="pink",command=check,font=("Verdana",12)).place(x=10,y=115); w.geometry("420x250"); w.mainloop()