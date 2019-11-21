"""
Common test configuration for all test subdirectories.
Put here only those things that can not be done through command line options and pytest.ini file.
"""



#https://stackoverflow.com/questions/37563396/deleting-py-test-tmpdir-directory-after-successful-test-case
# @pytest.fixture(scope='session')
# def temporary_dir(tmpdir_factory):
#     img = compute_expensive_image()
#     fn = tmpdir_factory.mktemp('data').join('img.png')
#     img.save(str(fn))
#     return fn