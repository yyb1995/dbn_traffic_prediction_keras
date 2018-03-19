import xlrd
import os
import numpy as np

def AbileneProcess(inputfilename, outputloc, routernum, datatype):
    """
    :param inputfilename: imput a xlsx data file
    :param outputloc: output file location
    :param routernum: output router num
    :param datatype:
        1: <realOD>
        2: <simpleGravityOD>
        3: <simpleTomogravityOD>
        4: <generalGravityOD>
        5: <generalTomogravityOD>
    :return: none
    """
    data_raw = xlrd.open_workbook(inputfilename)
    table = data_raw.sheets()[0]
    data_extract = []
    data_output = []
    if datatype in [1,2,3,4,5]:
        for i in range(datatype - 1,720,5):
            data_extract.append(table.col_values(i))
    else:
        print('Wrong type choice!')
    data_extract = np.reshape(data_extract,(144,2016))
    for i in range(routernum):
        for j in range(routernum):
            data_output.append(data_extract[i * 12 + j,:])
    data_output = np.reshape(data_output,(routernum * routernum,2016))
    path1 = outputloc
    path2 = os.path.splitext(os.path.split(inputfilename)[1])[0];
    path = path1 + '\\' + path2 + '_' + str(routernum * routernum) +\
           '_' + str(datatype) + '.npy'
    np.save(path, data_output)
    print('Process successfully!')

if __name__ == "__main__":
    AbileneProcess('E:\\北航\\研究生\\Abilene TM\\Abilene_xlsx\\20040529.xlsx', 'E:\\北航\\研究生\\Abilene TM\\Abilene_npy',
                   routernum = 12, datatype = 2)