from parameters import *
from datahelper import data_import


def input_data_path():
    '''
            Input data path
    '''
    data_path = input("Input the path to data file: ")
    print("Using data: " + data_path)
    if_transfer = input("Whether to use transfer: [Y/n] ")
    if (if_transfer == 'n' or if_transfer == 'N'):
        if_transfer = False
        print("Transfer mode: OFF")
    else:
        if_transfer = True
        print("Transfer mode: ON")
    data = data_import(data_path, transfer=if_transfer)
    return data


def input_glass_info():
    '''
            Input information about glass
    '''
    thickness = input("Input the thickness(mm) of glass: ")
    return float(thickness)*1e-3


def input_gst_like_info():
    '''
            Input information about GST
    '''
    data_type = input(
        "Choose the data type: 1.AM 2.CR (Now work on 1500-1600nm only) ")
    if (data_type == '1'):
        state = "AM"
    elif (data_type == '2'):
        state = "CR"
    else:
        raise ValueError("No such data type")
    print("Material phase: " + state)
    gst_thick = input("Input the thickness(nm): ")
    # Not sure it will work fine.
    # if (gst_thick != "20" and gst_thick != "80"):
    #     raise NotImplementedError("Now support 20nm and 80nm only")
    gst_thick = int(gst_thick)
    return state, gst_thick


def choose_data_type():
    '''
            User interface function
    '''
    material = input(
        "The material to test: 1.glass 2.GST 3.GeTe 4.AIST 5.others ")
    if (material == '1'):
        data = input_data_path()
        init_dict = [1.52909, 1.52769]
        thick = input_glass_info()
        data_name = "GLASS"
    elif (material == '2'):
        data = input_data_path()
        state, thick = input_gst_like_info()
        data_dict = {"AM20": GST_AM_20, "CR20": GST_CR_20, "CR80": GST_CR_80}
        init_dict = data_dict[state+str(thick)]
        data_name = "GST"
    elif (material == "3"):
        data = input_data_path()
        state, thick = input_gst_like_info()
        data_dict = {"AM": GeTe_AM, "CR": GeTe_CR}
        init_dict = data_dict[state]
        data_name = "GeTe"
    elif (material == "4"):
        data = input_data_path()
        state, thick = input_gst_like_info()
        data_dict = {"AM": AIST_AM, "CR": AIST_CR}
        init_dict = data_dict[state]
        data_name = "AIST"
    else:
        raise NotImplementedError("Material not support now，we will work hard :)")

    return data, init_dict, thick, data_name
