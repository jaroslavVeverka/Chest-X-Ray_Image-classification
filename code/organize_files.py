from zipfile import ZipFile


source_path = 'C:/Users/jarda/OneDrive/Documents/archive.zip'
destination_path = 'C:/Users/jarda/IdeaProjects/Chest X-Ray_Image classification/images'

with ZipFile(source_path, 'r') as zipObj:
    zipObj.extractall(destination_path)

print('Directory successfully unzipped in to destination directory.')
