import h5py
file=h5py.File( 'haloprops_all.h5', 'r')
for snapshot in file:
    grp = file[snapshot]
    print('group attributes ', snapshot)
    for att in grp.attrs.items():
        print(att)
    print(list(grp.keys()))
    print(grp['mvir'][:])
file.close()
