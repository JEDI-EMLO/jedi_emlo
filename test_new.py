import os
import pytest


def test_datazip():
    pass_this = True
    mydir = os.getcwd()
    if 'data.zip' in os.listdir(mydir):
        pass_this = False
        pass
    assert pass_this == True, "data.zip file exists. Remove it before pushing it to github"


def test_model():
    pass_this = True
    mydir = os.getcwd()
    if 'model.pth' in os.listdir(mydir):
        pass_this = False
        pass
    assert pass_this == True, "model.pth file exists. Remove it before pushing it to github"

