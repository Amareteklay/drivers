# cause-effect

I cloned this repo from Amare's github to work on my local computer. 
https://github.com/Amareteklay/cause-effect


print("Ege learning Python")
print("Ege created an environment from this link:  https://www.youtube.com/watch?v=D2cwvpJSBX4" )

comment = "Terminal REPL is a great way to learn Python"
# run this at the terminal. 
#ollama pull llama3.2

# try to run Amare's script
# cd go to cause-effect folder

git pull origin main # pull from main branch always as a first thing
#install requirements
pip3 install -r requirements.txt

#
pip3 install streamlit
streamlit run app.py


# git comments
git add . # do everything, add all the changes in my local branch
git commit -m "message" # commit with a message
git push origin main # push to main branch
git branch # check branches

git checkout -b ege_branch # create a new branch
git push origin ege_branch
git status # check status

git reset --hard ### this is to do the rest so git pull would work. 

git push origin ege_branch
