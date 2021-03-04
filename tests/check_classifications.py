import cocpit
import pandas as pd
import matplotlib.pyplot as plt

desired_size = 1000
df = pd.read_sql_table('2000_ARM', 'sqlite:///saved_models/2000_ARM.db')  
print(df)
for file, class_ in zip(df['filename'], df['classification']):
    image = cocpit.pic.Image('cpi_data/campaigns/2000_ARM/single_imgs/', file)
    image.resize_stretch(desired_size)
    print(file, class_)
    plt.imshow(image.image_og)
    plt.show()
    plt.pause(11)