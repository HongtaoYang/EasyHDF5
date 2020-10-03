import click

from h5_object import HDFConnection


@click.group()
def main():
    pass


@main.command(help="Create new or update existing hdf5 file.")
@click.option("--filename", type=str, help="Name of the hdf5 file.")
@click.option("--image_root", type=str, help="Image root directories containing images to be put into the hdf5 file.")
@click.option("--insert_to", type=str, help="Where to insert data inside hdf5 file. defaults to root.", default="/")
def update(filename: str, image_root: str, insert_to: str):
    hdf = HDFConnection(filename)
    hdf.insert_directory(image_root, insert_to)
    hdf.close()



if __name__ == "__main__":
    main()