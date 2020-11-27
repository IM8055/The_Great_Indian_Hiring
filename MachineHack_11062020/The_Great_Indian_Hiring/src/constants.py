import os


constROOT_DIR = os.path.dirname(os.path.abspath(__file__))
constPROJECTPATH = os.path.dirname(constROOT_DIR)
constTRAINFILENAME = 'Train.csv'
constTESTFILENAME = 'Test.csv'
constLOGFILENAME = 'ModelLogs.txt'
constTARGETCOLUMN = 'UnitPrice'
constTWILIOACCOUNTSID = 'ACbfbf6195c8f5a7eccbe5ff4747d3647e'
constTWILIOACCOUNTAUTH = 'db337b0341cdb9031eee4387230bc55d'
constTWILIOWHATSAPPFROM = 'whatsapp:+14155238886'
constTWILIOWHATSAPPTO = 'whatsapp:+919840517151'

COLUMNSTODROP = ['InvoiceDate',
                 'f_GrpCustomerID_CustomerID_size',
                 'f_GrpCustomerID_Quantity_sum',
                 'f_GrpCustomerID_Quantity_min',
                 'f_GrpCustomerID_Quantity_max',
                 'f_GrpCustomerID_StockCode_nunique',
                 'f_GrpInvoiceNo_Quantity_sum',
                 'f_GrpInvoiceNo_Quantity_min',
                 'f_GrpInvoiceNo_Quantity_max',
                 # 'f_GrpInvoiceNo_Quantity_mean',
                 'f_GrpInvoiceNo_InvoiceNo_size',
                 'f_GrpInvoiceNo_StockCode_nunique',
                 # # The below 2 columns are same as 'f_GrpCustomerID_CustomerID_size
                 'f_GrpCustomerID_InvoiceNo_size',
                 'f_GrpCustomerID_StockCode_size',
                 # # Dropping 'f_GrpCustomerID_CustomerID_nunique' since all values are the same
                 'f_GrpCustomerID_CustomerID_nunique',
                 # The below column is same as 'f_GrpInvoiceNo_InvoiceNo_size'
                 'f_GrpInvoiceNo_StockCode_size',
                 # # 'f_GrpInvoiceNo_InvoiceNo_nunique' is always 1
                 'f_GrpInvoiceNo_InvoiceNo_nunique'
                 ]

