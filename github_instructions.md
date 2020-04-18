## Working with Github

Here I am listing a few simple Github commands to help you work with Github for the project.

* Copy the repository URL.
* Open a terminal in the folder where you want to have a local copy of the repository.
* To clone the repository - *git clone <repository_url>*
* You can list all branches by - *git branch -a*
* To check which branch you are currently on - *git branch*
* To go to your team branch - *git checkout <branch_name>*
* Make the changes you want to make in the local copy of the folder.
* Now when you check the status of your local copy of the repo, you should see uncommitted files if you made changes to any existing file and/or added/deleted any file - *git status*
* Add the files to the staging area by - *git add .* 
* Note that this adds all the changes you made to any of the files. To add changes made to a specific file, - *git add <file_name>*
* Now the files are ready to be committed. You can check the status (*git status*) again and you will see files ready for commit (in green)
* Now make the commit with a commit message - *git commit -m "<Commit_Message>"* 
* The commit message is basically to keep track of what changes you made in this particular commit. A good commit message is important. For example a message like "added mnist code" or "modularized main.py" is good enough.
* To push the changes to your branch - *git push origin <branch_name>*
* You should now be able to see the changes on GitHub
* Cheers!
