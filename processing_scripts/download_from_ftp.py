import ftplib
import glob
import os


def list_of_files(path, extension, recursive=False):
    """
    Return a list of filepaths for each file into path with the target extension.
    If recursive, it will loop over subfolders as well.
    """
    if not recursive:
        for file_path in glob.iglob(path + "/*." + extension):
            print(file_path)
    else:
        for root, dirs, files in os.walk(path):
            yield from glob.iglob(root + "/*." + extension)


path = "/pub/download/data/vprzy296853/"
extension = "roi"
ftp = ftplib.FTP("data.eol.ucar.edu")
ftp.login("anonymous", "vprzybylo@albany.edu")
ftp.cwd(path)
files = ftp.nlst("*.roi")
print(files)
for file in files:
    ftp.retrbinary("RETR " + file, open(file, "wb").write)
ftp.quit()
