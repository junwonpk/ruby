git filter-branch -f --env-filter '
echo $GIT_AUTHOR_EMAIL
OLD_EMAIL="john@Ruby2.nfwnhrqpw3tepcqoskly4pbn4e.jx.internal.cloudapp.net"
CORRECT_NAME="Junwon Park"
CORRECT_EMAIL="junwonpk@stanford.edu"
if [ "$GIT_COMMITTER_EMAIL" = "$OLD_EMAIL" ]
then
    export GIT_COMMITTER_NAME="$CORRECT_NAME"
    export GIT_COMMITTER_EMAIL="$CORRECT_EMAIL"
fi
if [ "$GIT_AUTHOR_EMAIL" = "$OLD_EMAIL" ]
then
    export GIT_AUTHOR_NAME="$CORRECT_NAME"
    export GIT_AUTHOR_EMAIL="$CORRECT_EMAIL"
fi
' --tag-name-filter cat -- --branches --tags