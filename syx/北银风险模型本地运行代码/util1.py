# encoding=utf8
import os
import pickle as pickle
import simplejson as json
import codecs
import smtplib # python3 自带
import pandas as pd
import numpy as np
import configparser
import psycopg2
from jinja2 import Template
from sqlalchemy import create_engine
# import pymysql
import time
import datetime 
from dateutil.relativedelta import relativedelta

import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] =['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] =False   #用来正常显示负号

from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import roc_curve, auc,recall_score, precision_score,f1_score,confusion_matrix,roc_auc_score



#*************************基础函数**********************************************
def read_mysql(sql):
    conn = pymysql.connect('192.168.56.4','root','123456','kg')
    df = pd.read_sql(sql=sql,con=conn)
    conn.close()
    print(df.shape)
    print(sql)
    return df


def load_data_from_pickle(file_path, file_name):
    """加载pkl文件"""
    file_path_name = os.path.join(file_path, file_name)
    with open(file_path_name, 'rb') as infile:
        result = pickle.load(infile)
    return result

def save_data_to_pickle(obj, file_path, file_name):
    """保存pkl文件"""
    file_path_name = os.path.join(file_path, file_name)
    with open(file_path_name, 'wb') as outfile:
        pickle.dump(obj, outfile)


#*************************数据读取**********************************************
class IndBondDataPrepare:
    """产业债相关数据查询
    """
    
    def __init__(self):
        pass 
    
    def get_entid_baseinfo(self):
        """获取全量发债主体和上市公司基本信息
        企业ID，企业名称，是否发债，是否上市，WIND行业分类1，WIND行业分类2，企业性质
        """
        sql = """SELECT *
            FROM p_user_info_v2 
            """  
        return read_mysql(sql)

    def get_windcode_baseinfo(self):
        """获取全量债券基本信息
        """
        sql = """SELECT * from p_windcode_info_v2
              """  
        return read_mysql(sql)

    def get_security_bond(self):
        """获取各个观察时点存续债券"""
        sql = """select * from p_cbond_security"""
        return read_mysql(sql)

    def get_default_all(self):
        """获取历史所有违约样本：WIND后台违约数据"""
        sql = """SELECT * 
            FROM p_bond_ent_default_date
            WHERE entdefaultfirstdate IS NOT NULL 
            """
        return read_mysql(sql)

    def get_default_all_v2(self):
        """获取历史所有违约样本：WIND终端违约数据"""
        sql = """SELECT * 
            FROM p_bond_ent_default_date_V2
            WHERE entdefaultfirstdate IS NOT NULL 
            """
        return read_mysql(sql)

    def get_security_entid(self):
        """获取各个观察时点有存续债券的主体"""
        sql = """SELECT distinct b.S_INFO_COMPCODE,b.S_INFO_COMPNAME,a.data_dt
                from p_cbond_security a
                LEFT JOIN p_windcode_info_v2 b
                ON a.s_info_windcode = b.s_info_windcode"""
        return read_mysql(sql) 


    def get_vars_widetable(self):
        """获取特征宽表"""
        sql = """
        SELECT  * from kg.a_industrial_debt_model_index_v1
        """
        return read_mysql(sql)

    def get_vars_rel(self):
        """获取特征宽表"""
        sql = """
        SELECT  * from kg.industrial_debt_control_link_risk_index
        """
        return read_mysql(sql)

    def get_vars_rel_by(self):
        """获取特征宽表"""
        sql = """
        SELECT  * from kg.a_industrial_debt_control_link_risk_index_0411
        """
        return read_mysql(sql)
    
    def get_bond_valuation_widetable(self):
        """获取特征宽表"""
        sql = """
        SELECT  * from kg.a_industrial_debt_bond_valuation_merge_index
        """
        return read_mysql(sql)

    def get_vars_info(self):
        """特征指标说明信息"""
        sql = """SELECT *
                FROM p_index_info"""
        vars_info = read_mysql(sql)
        vars_info['boc_index_code'] = vars_info['boc_index_code'].map(lambda x:x.upper())
        vars_info.columns = ['指标一级分类','指标二级分类','指标英文名称','指标中文名称','指标中文名称（中行）','指标英文名称（中行）','是否入模（中行）']  
    
        # 中英文字段名映射
        vars_dict=  {}
        for vars,row in vars_info.iterrows():
            vars_dict[row['指标英文名称']] = row['指标中文名称']  
        return vars_info,vars_dict

    def get_vars_reason(self):
        """指标阈值"""
        sql = """SELECT *
                FROM p_index_info_reason"""
        vars_info_reason = read_mysql(sql)
        vars_info_reason.columns = ['指标英文名称', '指标中文名称','阈值', '逻辑符号', '归因输出内容']  
 
        return vars_info_reason
    
    def get_BondIssuerREPORTPERIOD(self,entid,data_dt):
        sql = """-- 债券财务数据 观察时点最近的年报报告期
                SELECT  
                a.* 
                ,'{0}' AS DATA_DT
                FROM 
                (SELECT * ,row_number() over(partition by S_INFO_COMPCODE order by ANN_DT desc ) as row_num
                FROM kg.CBondBalanceSheet 
                WHERE REPORT_PERIOD like '%1231'
                AND STATEMENT_TYPE='408001000'
                AND ann_dt < '{0}' 
                AND ann_dt > replace(date_add('{0}', interval -3 year),'-','')
                AND AUDIT_AM = 2) a
                WHERE a.row_num <= 3
                AND S_INFO_COMPCODE = '{1}'
                """.format(data_dt,entid)
        df = read_mysql(sql).drop(['OBJECT_ID'],axis=1)
        if len(df) > 0:
            df['REPORT_PERIOD'] = df['REPORT_PERIOD'].astype(int)
            return str(df['REPORT_PERIOD'].max())
        else:
            return ''


    def get_AShareIssuerREPORTPERIOD(self,entid,data_dt):
        sql = """-- 股票财务数据 观察时点最近的年报报告期
                SELECT  
                a.* 
                ,'{0}' AS DATA_DT
                FROM 
                (SELECT * ,row_number() over(partition by S_INFO_COMPCODE order by ANN_DT desc ) as row_num
                FROM kg.AShareBalanceSheet 
                WHERE REPORT_PERIOD like '%1231'
                AND STATEMENT_TYPE='408001000'
                AND ann_dt < '{0}' 
                AND ann_dt > replace(date_add('{0}', interval -3 year),'-','')
                ) a
                WHERE a.row_num <= 3
                AND S_INFO_COMPCODE = '{1}'
                """.format(data_dt,entid)
        df = read_mysql(sql).drop(['OBJECT_ID'],axis=1)
        if len(df) > 0:
            df['REPORT_PERIOD'] = df['REPORT_PERIOD'].astype(int)
            return str(df['REPORT_PERIOD'].max())
        else:
            return ''


    def get_BondIssuer_Balance(self,entid,data_dt):
        sql = """-- 中国债券发行主体资产负债表 -最近3年距离观察点的年报
                SELECT  
                a.* 
                ,'{0}' AS DATA_DT
                FROM 
                (SELECT * ,row_number() over(partition by S_INFO_COMPCODE order by ANN_DT desc ) as row_num
                FROM kg.CBondBalanceSheet 
                WHERE REPORT_PERIOD like '%1231'
                AND STATEMENT_TYPE='408001000'
                AND ann_dt < '{0}' 
                AND ann_dt > replace(date_add('{0}', interval -3 year),'-','')
                AND AUDIT_AM = 2) a
                WHERE a.row_num <= 3
                AND S_INFO_COMPCODE = '{1}'
                """.format(data_dt,entid)
        return read_mysql(sql).drop(['OBJECT_ID'],axis=1)
    
    def get_BondIssuer_Income(self,entid,data_dt):
        sql = """-- 中国债券发行主体利润表 -最近3年距离观察点的年报
                SELECT  
                a.* 
                ,'{0}' AS DATA_DT
                FROM 
                (SELECT * ,row_number() over(partition by S_INFO_COMPCODE order by ANN_DT desc ) as row_num
                FROM kg.CBondIncome 
                WHERE REPORT_PERIOD like '%1231'
                AND STATEMENT_TYPE='408001000'
                AND ann_dt < '{0}' 
                AND ann_dt > replace(date_add('{0}', interval -3 year),'-','')
                AND AUDIT_AM = 2) a
                WHERE a.row_num <= 3
                AND S_INFO_COMPCODE = '{1}'
                """.format(data_dt,entid)
        return read_mysql(sql).drop(['OBJECT_ID'],axis=1)
    
    def get_BondIssuer_CashFlow(self,entid,data_dt):
        sql = """-- 中国债券发行主体现金流量表 -最近3年距离观察点的年报
                SELECT  
                a.* 
                ,'{0}' AS DATA_DT
                FROM 
                (SELECT * ,row_number() over(partition by S_INFO_COMPCODE order by REPORT_PERIOD desc ) as row_num
                FROM kg.CBondCashFlow 
                WHERE REPORT_PERIOD like '%1231'
                AND STATEMENT_TYPE='408001000'
                AND ann_dt < '{0}' 
                AND ann_dt > replace(date_add('{0}', interval -3 year),'-','')
                AND AUDIT_AM = 2) a
                WHERE a.row_num <= 3
                AND S_INFO_COMPCODE = '{1}'
                """.format(data_dt,entid)
        return read_mysql(sql).drop(['OBJECT_ID'],axis=1)

    
    def get_BondIssuer_Financial(self,entid,data_dt):
        sql = """-- 中国债券发行主体财务指标表 -最近3年距离观察点的年报
                SELECT  
                a.* 
                ,'{0}' AS DATA_DT
                FROM 
                (SELECT * ,row_number() over(partition by S_INFO_COMPCODE order by REPORT_PERIOD desc ) as row_num
                FROM kg.CBondFinancialIndicator 
                WHERE REPORT_PERIOD like '%1231'
                AND ann_dt < '{0}' 
                AND ann_dt > replace(date_add('{0}', interval -3 year),'-','')
                   ) a
                WHERE a.row_num <= 3
                AND S_INFO_COMPCODE = '{1}'
                """.format(data_dt,entid)
        return read_mysql(sql).drop(['OBJECT_ID'],axis=1)
    
    def get_AShareIssuer_Balance(self,entid,data_dt):
        sql = """-- 股票发行发行主体资产负债表 -最近3年距离观察点的年报
                SELECT  
                a.* 
                ,'{0}' AS DATA_DT
                FROM 
                (SELECT * ,row_number() over(partition by S_INFO_COMPCODE order by REPORT_PERIOD desc ) as row_num
                FROM kg.AShareBalanceSheet 
                WHERE REPORT_PERIOD like '%1231'
                AND STATEMENT_TYPE='408001000'
                AND ann_dt < '{0}' 
                AND ann_dt > replace(date_add('{0}', interval -3 year),'-','')
                ) a
                WHERE a.row_num <= 3
                AND S_INFO_COMPCODE = '{1}'
                """.format(data_dt,entid)
        return read_mysql(sql).drop(['OBJECT_ID'],axis=1)
    
    def get_AShareIssuer_Income(self,entid,data_dt):
        sql = """-- 股票发行发行主体利润表 -最近3年距离观察点的年报
                SELECT  
                a.* 
                ,'{0}' AS DATA_DT
                FROM 
                (SELECT * ,row_number() over(partition by S_INFO_COMPCODE order by REPORT_PERIOD desc ) as row_num
                FROM kg.AShareIncome 
                WHERE REPORT_PERIOD like '%1231'
                AND STATEMENT_TYPE='408001000'
                AND ann_dt < '{0}' 
                AND ann_dt > replace(date_add('{0}', interval -3 year),'-','')
                ) a
                WHERE a.row_num <= 3
                AND S_INFO_COMPCODE = '{1}'
                """.format(data_dt,entid)
        return read_mysql(sql).drop(['OBJECT_ID'],axis=1)
    
    def get_AShareIssuer_CashFlow(self,entid,data_dt):
        sql = """-- 股票发行发行主体现金流量表 -最近3年距离观察点的年报
                SELECT  
                a.* 
                ,'{0}' AS DATA_DT
                FROM 
                (SELECT * ,row_number() over(partition by S_INFO_COMPCODE order by REPORT_PERIOD desc ) as row_num
                FROM kg.AShareCashFlow 
                WHERE REPORT_PERIOD like '%1231'
                AND STATEMENT_TYPE='408001000'
                AND ann_dt < '{0}' 
                AND ann_dt > replace(date_add('{0}', interval -3 year),'-','')
                ) a
                WHERE a.row_num <= 3
                AND S_INFO_COMPCODE = '{1}'
                """.format(data_dt,entid)
        return read_mysql(sql).drop(['OBJECT_ID'],axis=1)
    
    def get_AShareIssuer_Financial(self,entid,data_dt):
        sql = """-- 股票发行发行主体财务指标表 -距离观察点最近的3年年报
                SELECT  
                 a.* 
                ,'{0}' AS DATA_DT
                FROM 
                (SELECT * ,row_number() over(partition by S_INFO_COMPCODE order by REPORT_PERIOD desc ) as row_num
                FROM kg.AShareFinancialIndicator 
                WHERE REPORT_PERIOD like '%1231'
                AND ann_dt < '{0}' 
                AND ann_dt > replace(date_add('{0}', interval -3 year),'-','')
                )  a
                WHERE  a.row_num <= 3
                AND S_INFO_COMPCODE = '{1}'
                """.format(data_dt,entid)
        return read_mysql(sql).drop(['OBJECT_ID'],axis=1)
    
    def get_SecurityBondsOfEntid(self,entid,data_dt):
        """某个时点发债企业对应的存续债券名单"""
        sql = """ 
                SELECT DISTINCT S_INFO_WINDCODE
                FROM kg.p_windcode_info_v2
                WHERE S_INFO_COMPCODE  = '{0}'
                AND  WINDCODE_TYPE ='bond'
                and B_INFO_CARRYDATE < '{1}' 
                and B_INFO_PAYMENTDATE > '{1}'
                """.format(entid,data_dt)
        df = read_mysql(sql)
        if len(df) > 0:
            return df['S_INFO_WINDCODE'].tolist() 
        else:
            return []

    def get_EntidByBond(self,windcode):
        """根据债券ID，获取对应企业ID"""
        sql = """SELECT DISTINCT S_INFO_COMPCODE
            FROM p_windcode_info_v2
            WHERE S_INFO_WINDCODE = '{}'
          """.format(windcode)
    
        return read_mysql(sql)['S_INFO_COMPCODE'][0]

    def get_EntnameByEntid(self,entid):
        """根据企业ID，获取企业名称"""
        sql = """SELECT COMP_ID AS S_INFO_COMPCODE ,COMP_NAME AS S_INFO_COMPNAME
            FROM CompIntroduction
            WHERE COMP_ID = '{0}'""".format(entid)
        df = read_mysql(sql)
        entid_entname_dict = {}
        for index,row in df.iterrows():
            entid_entname_dict['S_INFO_COMPCODE'] = row['S_INFO_COMPNAME']

        return entid_entname_dict['S_INFO_COMPCODE']

    def get_EntidByEntname(self,entname):
        """根据企业名称，获取企业ID"""
        sql = """SELECT COMP_ID AS S_INFO_COMPCODE ,COMP_NAME AS S_INFO_COMPNAME
            FROM CompIntroduction
            WHERE COMP_NAME = '{0}'""".format(entname)
        df = read_mysql(sql)
        entid_entname_dict = {}
        for index,row in df.iterrows():
            entid_entname_dict['S_INFO_COMPNAME'] = row['S_INFO_COMPCODE']
        return entid_entname_dict['S_INFO_COMPNAME']
    
    def get_ShareWindcodeByEntid(self,entid):
        """根据企业ID，获取股票代码"""
        sql = """SELECT DISTINCT S_INFO_WINDCODE
            FROM AShareDescription
            WHERE S_INFO_COMPCODE = '{}'
          """.format(entid)
        df = read_mysql(sql)
        if len(df)>0:
            return df['S_INFO_WINDCODE'][0]
        else:
            None

    def get_BondInfoByWindcode(self,windcode):
        """根据债券编号，获取债券相关信息"""
        sql = """SELECT *
            FROM cbonddescription
            WHERE S_INFO_WINDCODE = '{0}'""".format(windcode)
        return read_mysql(sql).drop(['OBJECT_ID'],axis=1).to_dict('record')[0]
    
    def get_CompInfoByEntid(self,entid):
        """根据企业ID，获取企业基本信息"""
        sql = """SELECT *
            FROM CompIntroduction
            WHERE COMP_ID = '{0}'""".format(entid)
        return read_mysql(sql).drop(['OBJECT_ID'],axis=1).to_dict('record')[0]
    
    def get_AshareInfoByEntid(self,entid):
        """根据企业ID，获取股票基本信息"""
        sql = """SELECT *
            FROM AShareDescription
            WHERE S_INFO_COMPCODE = '{0}'""".format(entid)
        return read_mysql(sql).drop(['OBJECT_ID'],axis=1).to_dict('record')[0]
    
    def get_BondAmountByWindcode(self,windcode,data_dt):
        """根据债券ID和时点，获取债券份额"""
        sql = """SELECT S_INFO_WINDCODE,B_INFO_OUTSTANDINGBALANCE
            FROM (      
            SELECT *,row_number() over(partition by S_INFO_WINDCODE  order by S_INFO_ENDDATE desc)   as rn
            FROM cbondamount
            WHERE s_info_windcode = '{0}'
            AND S_INFO_ENDDATE < '{1}'
            )t
            WHERE t.rn = 1""".format(windcode,data_dt) 
        return read_mysql(sql)['B_INFO_OUTSTANDINGBALANCE'][0]
    
    
    def get_CreditRatingMap(self):
        """评级符号和数字映射表"""
        sql = """
            SELECT  CODE AS CREDITRATING,VALUE AS CREDITRATING_VALUE
            FROM code_value_table
            WHERE table_name = 'csiimplicitdefaultrate'
            and column_name = 'csi_creditrating'
            """
        return read_mysql(sql)
    
    def get_BondDefaultFirstDate(self):
        """获取债券首次违约日期"""
        sql = """SELECT B_INFO_WINDCODE AS S_INFO_WINDCODE,min(B_DEFAULT_DATE) AS BondDefaultFirstDate
                FROM CBondDefaultReportform
                GROUP BY B_INFO_WINDCODE"""
        return read_mysql(sql)

    def get_SingleBondDefaultFirstDate(self,windcode):
        """获取债券首次违约日期"""
        sql = """SELECT B_INFO_WINDCODE AS S_INFO_WINDCODE,min(B_DEFAULT_DATE) AS BondDefaultFirstDate
                FROM CBondDefaultReportform
                WHERE B_INFO_WINDCODE = '{}'
                GROUP BY B_INFO_WINDCODE""".format(windcode)
        df = read_mysql(sql)
        if len(df) > 0:
            return df['BondDefaultFirstDate'][0]
        else:
            return ''
        
    def get_EntDefaultFirstDate(self,entid):
        """根据企业ID，获取企业首次违约日期"""
        sql = """SELECT * 
                    FROM p_bond_ent_default_date
                    WHERE entdefaultfirstdate IS NOT NULL 
                    AND S_INFO_COMPCODE = '{}'""".format(entid)
        df = read_mysql(sql)
        if len(df) > 0:
            return df['EntDefaultFirstDate'][0]
        else:
            return ''
        
    def get_BondType(self,windcode):
        """根据债券ID，获取债券类别"""
        sql = """SELECT S_INFO_WINDCODE,S_INFO_INDUSTRYNAME
                FROM cbondindustrywind
                WHERE S_INFO_WINDCODE = '{}'
                """.format(windcode)
        return read_mysql(sql)['S_INFO_INDUSTRYNAME'][0]
    

    def get_BondEntIndustryProperty(self,entid):
        """根据企业ID，获取WIND行业分类和企业性质"""
        sql = """SELECT DISTINCT S_INFO_COMPCODE,S_INFO_COMPIND_NAME1,B_AGENCY_GUARANTORNATURE
                    from cbondissuer
                    WHERE S_INFO_COMPCODE = '{}'""".format(entid)
        df = read_mysql(sql)
        return df['S_INFO_COMPIND_NAME1'][0],df['B_AGENCY_GUARANTORNATURE'][0]
    
    def get_CsiCreditRating(self,windcode):
        """根据债券ID，获取中债隐含评级全量数据"""
        sql = """SELECT *
                from kg.csiimplicitdefaultrate
                WHERE s_info_windcode = '{}'""".format(windcode)
        df1 = read_mysql(sql)
        df2 = self.get_CreditRatingMap()
        df = pd.merge(df1,df2,how='left',left_on='CSI_CREDITRATING',right_on='CREDITRATING')
        df = df.drop(['OBJECT_ID','CREDITRATING'],axis=1).rename(columns={'CREDITRATING_VALUE':'CSI_CREDITRATING_VALUE'})
        return df
    
    def plot_CsiCreditTrend(self,windcode,file_path=None):
        """根据债券ID，绘制中债隐含评级变动趋势"""
        df = self.get_CsiCreditRating(windcode)
        if len(df) > 0:
            df1 = df.copy()
            df1 = df1[df1['TRADE_DT'].notna()]
            df1 = df1.sort_values(by=['TRADE_DT'])
            df1 = df1[df1['CSI_CREDITRATING_VALUE'].notna()]
            df1['CSI_CREDITRATING_VALUE_AFTER'] = df1['CSI_CREDITRATING_VALUE'].shift(-1)
            df1 = df1[df1['CSI_CREDITRATING_VALUE_AFTER']!=df1['CSI_CREDITRATING_VALUE']]
            df1 = df1.reset_index(drop=True)


            entid = self.get_EntidByBond(windcode)  # 企业ID
            entname = self.get_EntnameByEntid(entid) # 企业名称

            bond_info_dict = self.get_BondInfoByWindcode(windcode)
            B_INFO_CARRYDATE = bond_info_dict['B_INFO_CARRYDATE']  # 债券计息起始日
            B_INFO_PAYMENTDATE = bond_info_dict['B_INFO_PAYMENTDATE'] # 债券计息兑付日
            B_ISSUE_FIRSTISSUE = bond_info_dict['B_ISSUE_FIRSTISSUE']  # 发行起始日
            BOND_RATING = bond_info_dict['BOND_RATING']  # 发行信用评级

            bond_type = self.get_BondType(windcode) # 债券类别
            S_INFO_COMPIND_NAME1,B_AGENCY_GUARANTORNATURE = self.get_BondEntIndustryProperty(entid) # 发债企业WIND行业和企业属性
            bond_default_date = self.get_SingleBondDefaultFirstDate(windcode) # 债券首次违约日期
            fig,ax = plt.subplots(figsize=(16,10))
            sns.barplot(x='TRADE_DT',y='CSI_CREDITRATING_VALUE',data=df1,ax=ax,palette='Blues_d')
            plt.ylabel('中债隐含评级')
            plt.xlabel("评级日期")
            plt.title('中债隐含评级变动趋势:{0} {2}({6},{7}) \n 债券违约日期:{1},发行评级:{8},计息起始日:{3},兑付日:{4},债券类别:{5}'
                    .format(windcode,bond_default_date,entname,B_INFO_CARRYDATE,B_INFO_PAYMENTDATE,
                            bond_type,S_INFO_COMPIND_NAME1,B_AGENCY_GUARANTORNATURE,BOND_RATING))
            # plt.gca().set_yticklabels(df1['CSI_CREDITRATING'].tolist())
            xticks = ax.get_xticks()
            for i in range(len(df1)):
                xy = (xticks[i],df1['CSI_CREDITRATING_VALUE'][i]*1.02)
                s = str(df1['CSI_CREDITRATING'][i])
                ax.annotate(
                    s = s,
                    xy = xy,
                    fontsize = 12,
                    color = "blue",
                    ha = "center",
                    va = "baseline"
                )
            if len(df1) > 20:
                plt.xticks(rotation=90) # 刻度旋转
            
            if file_path:
                plt.savefig(os.path.join(file_path,r'{}_{}.jpg'.format(entname,windcode)))
            return plt.show()
        else:
            print('无数据')
    
    def get_BondCredit(self,windcode):
        """根据债券ID，获取债券信用评级全量数据"""
        sql = """SELECT S_INFO_WINDCODE,ANN_DT,B_INFO_CREDITRATING AS CREDITRATING,
            B_INFO_PRECREDITRATING,B_CREDITRATING_CHANGE
            ,B_RATE_STYLE,B_INFO_CREDITRATINGAGENCY
            FROM CBondRating
            WHERE  S_INFO_WINDCODE  = '{}'
            """.format(windcode)
        df1 = read_mysql(sql)
        df2 = self.get_CreditRatingMap()
        df = pd.merge(df1,df2,how='left',left_on='CREDITRATING',right_on='CREDITRATING')
        df = df[['S_INFO_WINDCODE','ANN_DT','CREDITRATING','CREDITRATING_VALUE']]
        return df

    def plot_BondCreditTrend(self,windcode,file_path):
        """根据债券ID，绘制债券信用评级变动趋势"""
        df = self.get_BondCredit(windcode)
        if len(df)>0:
            df1 = df.copy()
            df1 = df1[df1['ANN_DT'].notna()]
            df['ANN_DT'] = df['ANN_DT'].astype(int)
            df1 = df1.sort_values(by=['ANN_DT'])
            df1 = df1[df1['CREDITRATING_VALUE'].notna()]
            df1['CREDITRATING_VALUE_AFTER'] = df1['CREDITRATING_VALUE'].shift(-1)
            df1 = df1[df1['CREDITRATING_VALUE_AFTER']!=df1['CREDITRATING_VALUE']]
            df1 = df1.reset_index(drop=True)


            entid = self.get_EntidByBond(windcode)  # 企业ID
            entname = self.get_EntnameByEntid(entid) # 企业名称

            bond_info_dict = self.get_BondInfoByWindcode(windcode)
            B_INFO_CARRYDATE = bond_info_dict['B_INFO_CARRYDATE']  # 债券计息起始日
            B_INFO_PAYMENTDATE = bond_info_dict['B_INFO_PAYMENTDATE'] # 债券计息兑付日
            B_ISSUE_FIRSTISSUE = bond_info_dict['B_ISSUE_FIRSTISSUE']  # 发行起始日
            BOND_RATING = bond_info_dict['BOND_RATING']  # 发行信用评级

            bond_type = self.get_BondType(windcode) # 债券类别
            S_INFO_COMPIND_NAME1,B_AGENCY_GUARANTORNATURE = self.get_BondEntIndustryProperty(entid) # 发债企业WIND行业和企业属性
            bond_default_date = self.get_SingleBondDefaultFirstDate(windcode) # 债券首次违约日期
            fig,ax = plt.subplots(figsize=(16,10))
            sns.barplot(x='ANN_DT',y='CREDITRATING_VALUE',data=df1,ax=ax,palette='Blues_d')
            plt.ylabel('债券信用评级')
            plt.xlabel("评级日期")
            plt.title('债券信用评级变动趋势:{0} {2}({6},{7}) \n 债券违约日期:{1},发行评级:{8},计息起始日:{3},兑付日:{4},债券类别:{5}'
                    .format(windcode,bond_default_date,entname,B_INFO_CARRYDATE,B_INFO_PAYMENTDATE,
                            bond_type,S_INFO_COMPIND_NAME1,B_AGENCY_GUARANTORNATURE,BOND_RATING))
            # plt.gca().set_yticklabels(df1['CREDITRATING'].tolist())
            xticks = ax.get_xticks()
            for i in range(len(df1)):
                xy = (xticks[i],df1['CREDITRATING_VALUE'][i]*1.02)
                s = str(df1['CREDITRATING'][i])
                ax.annotate(
                    s = s,
                    xy = xy,
                    fontsize = 12,
                    color = "blue",
                    ha = "center",
                    va = "baseline"
                )
            if len(df1) > 20:
                plt.xticks(rotation=90) # 刻度旋转

            if file_path:
                plt.savefig(os.path.join(file_path,r'{}_{}.jpg'.format(entname,windcode)))
            return plt.show()
        else:
            print('无数据')
    
    def get_BondEntCredit(self,entid):
        """根据企业ID，获取主体信用评级全量数据"""
        sql = """SELECT DISTINCT S_INFO_COMPCODE,ANN_DT,B_INFO_CREDITRATING AS CREDITRATING
        FROM cbondissuerrating
        WHERE S_INFO_COMPCODE = '{}'
        AND  B_INFO_CREDITRATINGAGENCY in ('2','5','4','6','14','19','7','13','15')
        ORDER BY ANN_DT
            """.format(entid)
        df1 = read_mysql(sql)
        df1 = df1.drop_duplicates(['ANN_DT'])
        df2 = self.get_CreditRatingMap()
        df = pd.merge(df1,df2,how='left',left_on='CREDITRATING',right_on='CREDITRATING')
        df = df[['S_INFO_COMPCODE','ANN_DT','CREDITRATING','CREDITRATING_VALUE']]
        return df

    def plot_BondEntCreditTrend(self,entid,file_path):
        """根据企业ID，绘制主体信用评级变动趋势"""
        df = self.get_BondEntCredit(entid)
        if len(df) > 0:
            df1 = df.copy()
            df1 = df1[df1['ANN_DT'].notna()]
            df['ANN_DT'] = df['ANN_DT'].astype(int)
            df1 = df1.sort_values(by=['ANN_DT'])
            df1 = df1[df1['CREDITRATING_VALUE'].notna()]
            df1['CREDITRATING_VALUE_AFTER'] = df1['CREDITRATING_VALUE'].shift(-1)
            df1 = df1[df1['CREDITRATING_VALUE_AFTER']!=df1['CREDITRATING_VALUE']]
            df1 = df1.reset_index(drop=True)


            entname = self.get_EntnameByEntid(entid) # 企业名称

            S_INFO_COMPIND_NAME1,B_AGENCY_GUARANTORNATURE = self.get_BondEntIndustryProperty(entid) # 发债企业WIND行业和企业属性
            entid_default_date = self.get_EntDefaultFirstDate(entid) # 主体首次违约日期

            fig,ax = plt.subplots(figsize=(16,10))
            sns.barplot(x='ANN_DT',y='CREDITRATING_VALUE',data=df1,ax=ax,palette='Blues_d')
            plt.ylabel('主体信用评级')
            plt.xlabel("评级日期")
            plt.title('主体信用评级变动趋势:{0}({1},{2}),主体违约日期:{3}'
                    .format(entname,S_INFO_COMPIND_NAME1,B_AGENCY_GUARANTORNATURE,entid_default_date))
            # plt.gca().set_yticklabels(df1['CREDITRATING'].tolist())
            xticks = ax.get_xticks()
            for i in range(len(df1)):
                xy = (xticks[i],df1['CREDITRATING_VALUE'][i]*1.02)
                s = str(df1['CREDITRATING'][i])
                ax.annotate(
                    s = s,
                    xy = xy,
                    fontsize = 12,
                    color = "blue",
                    ha = "center",
                    va = "baseline"
                )
            if len(df1) > 20:
                plt.xticks(rotation=90) # 刻度旋转

            if file_path:
                plt.savefig(os.path.join(file_path,r'{}.jpg'.format(entname)))
            return plt.show()
        else:
            print('无数据')
    
    def get_TOTSHRByWindcode(self,entid,data_dt):
        windcode =  self.get_ShareWindcodeByEntid(entid)
        if windcode:
            """根据企业ID，获取股票总股本(万股)"""
            sql = """SELECT *
                FROM (
                SELECT
                S_INFO_WINDCODE
                ,TOT_SHR
                ,ROW_NUMBER() over(PARTITION BY S_INFO_WINDCODE  ORDER BY CHANGE_DT DESC) rn 
                FROM asharecapitalization
                WHERE CHANGE_DT < '{0}'
                AND S_INFO_WINDCODE = '{1}'
                ) t
                WHERE t.rn=1""".format(data_dt,windcode)
            df = read_mysql(sql)

            if len(df) > 0:
                return str(df['TOT_SHR'][0])
            else:
                print('请检查该企业是否是上市公司')
                return ''
        else:
            return ''
        
    def get_ShareFroNum(self,entid,data_dt):
        """根据企业ID，获取股票冻结总数量"""
        sql = """SELECT S_INFO_COMPCODE,ANN_DATE,S_FRO_BGDATE,S_FRO_ENDDATE,DISFROZEN_TIME,S_FRO_SHARES
                ,S_HOLDER_ID,S_HOLDER_NAME
                FROM ashareequfroinfo
                WHERE S_INFO_COMPCODE = '{}'
                AND (S_FRO_ENDDATE = '' OR S_FRO_ENDDATE IS NULL)
                AND (DISFROZEN_TIME = '' OR DISFROZEN_TIME IS NULL)
                AND S_FRO_BGDATE < '{}'""".format(entid,data_dt)

        df = read_mysql(sql)
        df = df[df['S_FRO_SHARES'].notna()]
        df['S_FRO_SHARES'] = df['S_FRO_SHARES'].astype(float)
        if len(df) > 0:
            return str(df['S_FRO_SHARES'].sum())
        else:
            return ''
        
    def get_ShareFroRatio(self,entid,data_dt):
        """根据企业ID，获取股票冻结比例"""
        TOTSHR = self.get_TOTSHRByWindcode(entid,data_dt)
        ShareFroNum = self.get_ShareFroNum(entid,data_dt)
        if TOTSHR == '' or ShareFroNum == '':
            return str(0)
        else:
            return str(round(float(ShareFroNum)/float(TOTSHR),4))
        
    def get_SharePledgeRatioByEntid(self,entid,data_dt):
        """根据企业ID，获取股票质押比例"""
        windcode =  self.get_ShareWindcodeByEntid(entid)
        if windcode:
            sql = """SELECT
                S_INFO_WINDCODE
                ,S_ENDDATE
                ,S_PLEDGE_RATIO
                FROM (
                SELECT *
                ,ROW_NUMBER() over(PARTITION BY S_INFO_WINDCODE ORDER BY S_ENDDATE DESC) rn
                FROM asharepledgeproportion
                WHERE S_INFO_WINDCODE = '{}'
                AND S_ENDDATE < '{}'
                ) t
                WHERE t.rn = 1""".format(windcode,data_dt)
            df = read_mysql(sql)
            if len(df) > 0:
                return str(df['S_PLEDGE_RATIO'][0])
            else:
                return str(0)
        else:
            return str(0)

    def get_ShareSuspensionByEntid(self,entid,data_dt):
        """根据企业ID，获取股票停牌相关特征"""

        windcode =  self.get_ShareWindcodeByEntid(entid)
        if windcode:
            sql = """-- 股票近2年累计停牌天数
                    SELECT
                    S_INFO_WINDCODE
                    ,count(distinct S_DQ_SUSPENDDATE) as SUMCUM_SUSPENDDAYS_2Y
                    FROM asharetradingsuspensionderived
                    WHERE S_DQ_SUSPENDDATE < '{0}'
                    AND S_INFO_WINDCODE = '{1}'
                    AND S_DQ_SUSPENDDATE > replace(date_add('{0}', INTERVAL -24 MONTH),'-','')
                    GROUP BY S_INFO_WINDCODE""".format(data_dt,windcode)
            df1 = read_mysql(sql)

            sql = """-- 股票近2年最长停牌天数
                    SELECT
                    S_INFO_WINDCODE
                    ,max(S_DQ_SUSPENDDAYS_CONTINU) as MAX_SUSPENDDAYS_2Y
                    FROM asharetradingsuspensionderived
                    WHERE S_DQ_SUSPENDDATE < '{0}'
                    AND S_INFO_WINDCODE = '{1}'
                    AND S_DQ_SUSPENDDATE > replace(date_add('{0}', INTERVAL -24 MONTH),'-','')
                    GROUP BY S_INFO_WINDCODE
                    """.format(data_dt,windcode)
            df2 = read_mysql(sql)

            sql = """-- 股票近2年累计停牌次数
                SELECT
                S_INFO_WINDCODE
                ,count(distinct S_DQ_SUSPENDDATE_START) AS COUNTCUM_SUSPENDDAYS_2Y
                FROM asharetradingsuspensionderived
                WHERE S_DQ_SUSPENDDATE < '{0}'
                AND S_INFO_WINDCODE = '{1}'
                AND S_DQ_SUSPENDDATE > replace(date_add('{0}', INTERVAL -24 MONTH),'-','')
                GROUP BY S_INFO_WINDCODE
                        """.format(data_dt,windcode)
            df3 = read_mysql(sql)

            dt = pd.DataFrame([windcode],columns=['S_INFO_WINDCODE'])
            df = pd.merge(dt,df1,on=['S_INFO_WINDCODE'],how='left')\
                    .merge(df2,on=['S_INFO_WINDCODE'],how='left')\
                    .merge(df3,on=['S_INFO_WINDCODE'],how='left')
            
            df = df.fillna(0)
            for col in ['SUMCUM_SUSPENDDAYS_2Y','MAX_SUSPENDDAYS_2Y','COUNTCUM_SUSPENDDAYS_2Y']:
                df[col] = df[col].astype(int)
            return df.to_dict('record')[0]
        else:
            return {}
    
    def get_SharePriceByEntid(self,entid,data_dt):
        """根据企业ID，获取近2年股价信息"""
        windcode =  self.get_ShareWindcodeByEntid(entid)
        if windcode:
            sql = """SELECT
                S_INFO_WINDCODE
                ,TRADE_DT
                ,S_DQ_ADJCLOSE
                ,S_DQ_ADJPRECLOSE
                ,S_DQ_VOLUME
                ,S_DQ_PCTCHANGE
                FROM ashareeodprices
                WHERE S_INFO_WINDCODE = '{0}'
                AND S_DQ_TRADESTATUS = '交易'
                AND TRADE_DT < '{1}'
                AND TRADE_DT > replace(date_add('{1}', INTERVAL -24 MONTH),'-','')
                ORDER BY  TRADE_DT""".format(windcode,data_dt)

            df = read_mysql(sql)
            return df
        else:
            return None

    def plot_SharePricePctChangeTrend(self,entid,data_dt,file_path):
        """根据企业ID，绘制股价涨跌幅近1年趋势图"""
        df = self.get_SharePriceByEntid(entid,data_dt)
        if df is not None:
            if len(df)>0:
                df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])
                entname = self.get_EntnameByEntid(entid) # 企业名称
                windcode = self.get_ShareWindcodeByEntid(entid)
                S_INFO_COMPIND_NAME1,B_AGENCY_GUARANTORNATURE = self.get_BondEntIndustryProperty(entid) # 发债企业WIND行业和企业属性
                entid_default_date = self.get_EntDefaultFirstDate(entid) # 主体首次违约日期
                df = df[df['TRADE_DT'] > pd.to_datetime(n_years_ago(start_date=data_dt[0:4] + '-' + data_dt[4:6] + '-' + data_dt[6:8],n=1))]
                fig,ax = plt.subplots(figsize=(16,10))
                sns.lineplot(x='TRADE_DT',y='S_DQ_PCTCHANGE',data=df,ax=ax,palette='Blues_d')
                plt.ylabel('涨跌幅%')
                plt.xlabel('交易日期')
                plt.title('股价涨跌幅变动趋势:{0}({1},{2}),主体违约日期:{3}'
                        .format(entname,S_INFO_COMPIND_NAME1,B_AGENCY_GUARANTORNATURE,entid_default_date))
                # plt.axvline(x=entid_default_date,c='red',ls='-')
                # plt.text(x=entid_default_date,y=df['S_DQ_PCTCHANGE'].median(),s='首次违约日期')
                # plt.axvline(x=data_dt,c='green',ls=':')
                # plt.text(x=data_dt,y=df['S_DQ_PCTCHANGE'].median()*10,s='观察日期')
                if file_path:
                    plt.savefig(os.path.join(file_path,r'{}_{}.jpg'.format(entname,windcode)))
                return plt.show()
            else:
                print('无数据')
        else:
            print('非上市公司')
    
    def plot_SharePriceTrend(self,entid,data_dt,file_path):
        """根据企业ID，绘制股价近1年趋势图"""
        df = self.get_SharePriceByEntid(entid,data_dt)
        if df is not None:
            if len(df) > 0:
                df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])
                entname = self.get_EntnameByEntid(entid) # 企业名称
                windcode = self.get_ShareWindcodeByEntid(entid)
                S_INFO_COMPIND_NAME1,B_AGENCY_GUARANTORNATURE = self.get_BondEntIndustryProperty(entid) # 发债企业WIND行业和企业属性
                entid_default_date = self.get_EntDefaultFirstDate(entid) # 主体首次违约日期
                df = df[df['TRADE_DT'] > pd.to_datetime(n_years_ago(start_date=data_dt[0:4] + '-' + data_dt[4:6] + '-' + data_dt[6:8],n=1))]
                fig,ax = plt.subplots(figsize=(16,10))
                sns.lineplot(x='TRADE_DT',y='S_DQ_ADJCLOSE',data=df,ax=ax,palette='Blues_d')
                plt.ylabel('收盘价')
                plt.xlabel('交易日期')
                plt.title('股价涨跌幅变动趋势:{0}({1},{2}),主体违约日期:{3}'
                        .format(entname,S_INFO_COMPIND_NAME1,B_AGENCY_GUARANTORNATURE,entid_default_date))
                # plt.axvline(x=entid_default_date,c='red',ls='-')
                # plt.text(x=entid_default_date,y=df['S_DQ_ADJCLOSE'].median()*0.98,s='首次违约日期')
                # plt.axvline(x=data_dt,c='green',ls=':')
                # plt.text(x=data_dt,y=df['S_DQ_ADJCLOSE'].median()*1.02,s='观察日期')

                if file_path:
                    plt.savefig(os.path.join(file_path,r'{}_{}.jpg'.format(entname,windcode)))
                return plt.show()
            else:
                print('无数据')
        else:
            print('非上市公司')
    
    def get_BondPriceByWindcode(self,windcode,data_dt):
        """根据债券ID，获取债券价格信息"""
        sql = """SELECT
                S_INFO_WINDCODE
                ,TRADE_DT
                ,B_DQ_ORIGINCLOSE
                ,B_DQ_VOLUME
                ,B_DQ_AMOUNT
                FROM p_cbondpricesnet_fs
                WHERE S_INFO_WINDCODE = '{0}'
                AND TRADE_DT < '{1}'
                AND TRADE_DT > replace(date_add('{1}', INTERVAL -24 MONTH),'-','')
                AND B_DQ_VOLUME > 0
                ORDER BY TRADE_DT
                """.format(windcode,data_dt)
        return read_mysql(sql)

    def plot_BondPriceTrend(self,windcode,data_dt,file_path):
        """根据债券ID，绘制债券价格近1年趋势图"""
        df = self.get_BondPriceByWindcode(windcode,data_dt)
        if len(df) > 0:
            df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])
            entid = self.get_EntidByBond(windcode)
            entname = self.get_EntnameByEntid(entid) # 企业名称
            
            bond_info_dict = self.get_BondInfoByWindcode(windcode)
            B_INFO_CARRYDATE = bond_info_dict['B_INFO_CARRYDATE']  # 债券计息起始日
            B_INFO_PAYMENTDATE = bond_info_dict['B_INFO_PAYMENTDATE'] # 债券计息兑付日
            B_ISSUE_FIRSTISSUE = bond_info_dict['B_ISSUE_FIRSTISSUE']  # 发行起始日
            BOND_RATING = bond_info_dict['BOND_RATING']  # 发行信用评级

            bond_type = self.get_BondType(windcode) # 债券类别
            S_INFO_COMPIND_NAME1,B_AGENCY_GUARANTORNATURE = self.get_BondEntIndustryProperty(entid) # 发债企业WIND行业和企业属性
            bond_default_date = self.get_SingleBondDefaultFirstDate(windcode) # 债券首次违约日期
            df = df[df['TRADE_DT'] > pd.to_datetime(n_years_ago(start_date=data_dt[0:4] + '-' + data_dt[4:6] + '-' + data_dt[6:8],n=1))]
            fig,ax = plt.subplots(figsize=(16,10))
            sns.lineplot(x='TRADE_DT',y='B_DQ_ORIGINCLOSE',data=df,ax=ax,palette='Blues_d')
            plt.ylabel('收盘价')
            plt.xlabel('交易日期')
            
            plt.title('债券价格趋势:{0} {2}({6},{7}) \n 债券违约日期:{1},发行评级:{8},计息起始日:{3},兑付日:{4},债券类别:{5}'
                    .format(windcode,bond_default_date,entname,B_INFO_CARRYDATE,B_INFO_PAYMENTDATE,
                            bond_type,S_INFO_COMPIND_NAME1,B_AGENCY_GUARANTORNATURE,BOND_RATING))

            # plt.axvline(x=bond_default_date,c='red',ls='-')
            # plt.text(x=bond_default_date,y=df['B_DQ_ORIGINCLOSE'].median(),s='首次违约日期')
            # plt.axvline(x=data_dt,c='green',ls=':')
            # plt.text(x=data_dt,y=df['B_DQ_ORIGINCLOSE'].median(),s='观察日期')
            if file_path:
                plt.savefig(os.path.join(file_path,r'{}_{}.jpg'.format(entname,windcode)))
            return plt.show()
        else:
            print('无数据')

    def get_EventInfoALL(self,data_dt):
        """根据企业ID，获取风险事件信息"""
        sql = """SELECT 
            DISTINCT COMPCODE AS S_INFO_COMPCODE
            ,PUBLISHDATE
            ,EVTTYPE1
            FROM riskwarningevents
            WHERE  PUBLISHDATE < '{0}'
            AND PUBLISHDATE > replace(date_add('{0}', INTERVAL -24 MONTH),'-','')
            ORDER BY PUBLISHDATE""".format(data_dt)
        return read_mysql(sql)
    
    def get_EventInfoByEntid(self,entid,data_dt):
        """根据企业ID，获取风险事件信息"""
        sql = """SELECT 
            DISTINCT COMPCODE AS S_INFO_COMPCODE
            ,PUBLISHDATE
            ,EVTTYPE1
            FROM riskwarningevents
            WHERE COMPCODE = '{0}'
            AND PUBLISHDATE < '{1}'
            AND PUBLISHDATE > replace(date_add('{1}', INTERVAL -24 MONTH),'-','')
            ORDER BY PUBLISHDATE""".format(entid,data_dt)
        return read_mysql(sql)


    def plot_EventTrend(self,entid,data_dt,file_path):
        """根据企业ID，绘制风险事件发生序列趋势图"""
        df = self.get_EventInfoByEntid(entid,data_dt)
        if len(df) > 0:
            df['eventid'] = np.arange(1,len(df)+1,1)

            df['PUBLISHDATE'] = pd.to_datetime(df['PUBLISHDATE'])
            entname = self.get_EntnameByEntid(entid) # 企业名称
            S_INFO_COMPIND_NAME1,B_AGENCY_GUARANTORNATURE = self.get_BondEntIndustryProperty(entid) # 发债企业WIND行业和企业属性
            entid_default_date = self.get_EntDefaultFirstDate(entid) # 主体首次违约日期
            df = df[df['PUBLISHDATE'] > pd.to_datetime(n_years_ago(start_date=data_dt[0:4] + '-' + data_dt[4:6] + '-' + data_dt[6:8],n=1))]
            fig,ax = plt.subplots(figsize=(16,10))
            sns.lineplot(x='PUBLISHDATE',y='eventid',data=df,ax=ax,palette='Blues_d')
            for index,row in df.iterrows():
                plt.text(x=row['PUBLISHDATE'],y=row['eventid'],s=row['EVTTYPE1'])

            plt.ylabel('事件序号')
            plt.xlabel('事件公告日期')
            plt.title('风险事件:{0}({1},{2}),主体违约日期:{3}'
                    .format(entname,S_INFO_COMPIND_NAME1,B_AGENCY_GUARANTORNATURE,entid_default_date))
            # plt.axvline(x=entid_default_date,c='red',ls='-')
            # plt.text(x=entid_default_date,y=df['eventid'].median()*1.2,s='首次违约日期')
            # plt.axvline(x=data_dt,c='green',ls=':')
            # plt.text(x=data_dt,y=df['eventid'].median(),s='观察日期')
            if file_path:
                plt.savefig(os.path.join(file_path,r'{}.jpg'.format(entname)))
            return plt.show()
        else:
            print('无数据!!!')
    
    def get_BondYTM(self,windcode,data_dt):
        sql = """SELECT
        S_INFO_WINDCODE
        ,TRADE_DT
        ,B_ANAL_YTM
        FROM p_cbond_valuation_fs
        WHERE S_INFO_WINDCODE = '{0}'
        AND TRADE_DT < '{1}'
        AND TRADE_DT > replace(date_add('{1}', INTERVAL -24 MONTH),'-','')
        AND B_ANAL_YTM <> 0
        ORDER BY TRADE_DT
        """.format(windcode,data_dt)
        return  read_mysql(sql)

    def plot_BondYTMTrend(self,windcode,data_dt,file_path):
        """根据债券ID，绘制债券到期收益率近1年趋势图"""
        df = self.get_BondYTM(windcode,data_dt)
        if len(df)>0:
            df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])
            entid = self.get_EntidByBond(windcode)
            entname = self.get_EntnameByEntid(entid) # 企业名称
            
            bond_info_dict = self.get_BondInfoByWindcode(windcode)
            B_INFO_CARRYDATE = bond_info_dict['B_INFO_CARRYDATE']  # 债券计息起始日
            B_INFO_PAYMENTDATE = bond_info_dict['B_INFO_PAYMENTDATE'] # 债券计息兑付日
            B_ISSUE_FIRSTISSUE = bond_info_dict['B_ISSUE_FIRSTISSUE']  # 发行起始日
            BOND_RATING = bond_info_dict['BOND_RATING']  # 发行信用评级

            bond_type = self.get_BondType(windcode) # 债券类别
            S_INFO_COMPIND_NAME1,B_AGENCY_GUARANTORNATURE = self.get_BondEntIndustryProperty(entid) # 发债企业WIND行业和企业属性
            bond_default_date = self.get_SingleBondDefaultFirstDate(windcode) # 债券首次违约日期
            df = df[df['TRADE_DT'] > pd.to_datetime(n_years_ago(start_date=data_dt[0:4] + '-' + data_dt[4:6] + '-' + data_dt[6:8],n=1))]
            fig,ax = plt.subplots(figsize=(16,10))
            sns.lineplot(x='TRADE_DT',y='B_ANAL_YTM',data=df,ax=ax,palette='Blues_d')
            plt.ylabel('到期收益率')
            plt.xlabel('交易日期')
            plt.title('债券到期收益率变动趋势:{0} {2}({6},{7}) \n 债券违约日期:{1},发行评级:{8},计息起始日:{3},兑付日:{4},债券类别:{5}'
                    .format(windcode,bond_default_date,entname,B_INFO_CARRYDATE,B_INFO_PAYMENTDATE,
                            bond_type,S_INFO_COMPIND_NAME1,B_AGENCY_GUARANTORNATURE,BOND_RATING))

            # plt.axvline(x=bond_default_date,c='red',ls='-')
            # plt.text(x=bond_default_date,y=df['B_ANAL_YTM'].median(),s='首次违约日期')
            # plt.axvline(x=data_dt,c='green',ls=':')
            # plt.text(x=data_dt,y=df['B_ANAL_YTM'].median(),s='观察日期')
            if file_path:
                plt.savefig(os.path.join(file_path,r'{}_{}.jpg'.format(entname,windcode)))
            return plt.show()
        else:
            print('无数据!!!')

    def get_BondAnaYield(self,windcode,data_dt):
        """根据债券ID，获取估价收益率，估价净价"""
        sql = """SELECT
        S_INFO_WINDCODE
        ,TRADE_DT
        ,B_ANAL_YIELD_CNBD
        ,B_ANAL_NET_CNBD
        FROM p_cbond_analysis_cnbd_fs
        WHERE S_INFO_WINDCODE = '{0}'
        AND TRADE_DT < '{1}'
        AND TRADE_DT > replace(date_add('{1}', INTERVAL -24 MONTH),'-','')
        AND B_ANAL_YIELD_CNBD <> 0
        ORDER BY TRADE_DT
        """.format(windcode,data_dt)
        return  read_mysql(sql)

    def plot_BondAnaYieldTrend(self,windcode,data_dt,file_path):
        """根据债券ID，绘制债券估价收益率近1年趋势图"""
        df = self.get_BondAnaYield(windcode,data_dt)
        if len(df)>0:
            df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])
            entid = self.get_EntidByBond(windcode)
            entname = self.get_EntnameByEntid(entid) # 企业名称
            
            bond_info_dict = self.get_BondInfoByWindcode(windcode)
            B_INFO_CARRYDATE = bond_info_dict['B_INFO_CARRYDATE']  # 债券计息起始日
            B_INFO_PAYMENTDATE = bond_info_dict['B_INFO_PAYMENTDATE'] # 债券计息兑付日
            B_ISSUE_FIRSTISSUE = bond_info_dict['B_ISSUE_FIRSTISSUE']  # 发行起始日
            BOND_RATING = bond_info_dict['BOND_RATING']  # 发行信用评级

            bond_type = self.get_BondType(windcode) # 债券类别
            S_INFO_COMPIND_NAME1,B_AGENCY_GUARANTORNATURE = self.get_BondEntIndustryProperty(entid) # 发债企业WIND行业和企业属性
            bond_default_date = self.get_SingleBondDefaultFirstDate(windcode) # 债券首次违约日期
            df = df[df['TRADE_DT'] > pd.to_datetime(n_years_ago(start_date=data_dt[0:4] + '-' + data_dt[4:6] + '-' + data_dt[6:8],n=1))]
            fig,ax = plt.subplots(figsize=(16,10))
            sns.lineplot(x='TRADE_DT',y='B_ANAL_YIELD_CNBD',data=df,ax=ax,palette='Blues_d')
            plt.ylabel('估价收益率')
            plt.xlabel('交易日期')
            plt.title('债券估价收益率变动趋势:{0} {2}({6},{7}) \n 债券违约日期:{1},发行评级:{8},计息起始日:{3},兑付日:{4},债券类别:{5}'
                    .format(windcode,bond_default_date,entname,B_INFO_CARRYDATE,B_INFO_PAYMENTDATE,
                            bond_type,S_INFO_COMPIND_NAME1,B_AGENCY_GUARANTORNATURE,BOND_RATING))

            # plt.axvline(x=bond_default_date,c='red',ls='-')
            # plt.text(x=bond_default_date,y=df['B_ANAL_YIELD_CNBD'].median(),s='首次违约日期')
            # plt.axvline(x=data_dt,c='green',ls=':')
            # plt.text(x=data_dt,y=df['B_ANAL_YIELD_CNBD'].median(),s='观察日期')
            if file_path:
                plt.savefig(os.path.join(file_path,r'{}_{}.jpg'.format(entname,windcode)))
            return plt.show()
        else:
            print('无数据!!!')

    def get_BondAnaNet(self,windcode,data_dt):
        """根据债券ID，获取估价收益率，估价净价"""
        sql = """SELECT
        S_INFO_WINDCODE
        ,TRADE_DT
        ,B_ANAL_YIELD_CNBD
        ,B_ANAL_NET_CNBD
        FROM p_cbond_analysis_cnbd_fs
        WHERE S_INFO_WINDCODE = '{0}'
        AND TRADE_DT < '{1}'
        AND TRADE_DT > replace(date_add('{1}', INTERVAL -24 MONTH),'-','')
        AND B_ANAL_YIELD_CNBD <> 0
        ORDER BY TRADE_DT
        """.format(windcode,data_dt)
        return  read_mysql(sql)

    def plot_BondAnaNetTrend(self,windcode,data_dt,file_path):
        """根据债券ID，绘制债券估价净价近1年趋势图"""
        df = self.get_BondAnaNet(windcode,data_dt)
        if len(df)>0:
            df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])
            entid = self.get_EntidByBond(windcode)
            entname = self.get_EntnameByEntid(entid) # 企业名称
            
            bond_info_dict = self.get_BondInfoByWindcode(windcode)
            B_INFO_CARRYDATE = bond_info_dict['B_INFO_CARRYDATE']  # 债券计息起始日
            B_INFO_PAYMENTDATE = bond_info_dict['B_INFO_PAYMENTDATE'] # 债券计息兑付日
            B_ISSUE_FIRSTISSUE = bond_info_dict['B_ISSUE_FIRSTISSUE']  # 发行起始日
            BOND_RATING = bond_info_dict['BOND_RATING']  # 发行信用评级

            bond_type = self.get_BondType(windcode) # 债券类别
            S_INFO_COMPIND_NAME1,B_AGENCY_GUARANTORNATURE = self.get_BondEntIndustryProperty(entid) # 发债企业WIND行业和企业属性
            bond_default_date = self.get_SingleBondDefaultFirstDate(windcode) # 债券首次违约日期
            df = df[df['TRADE_DT'] > pd.to_datetime(n_years_ago(start_date=data_dt[0:4] + '-' + data_dt[4:6] + '-' + data_dt[6:8],n=1))]
            fig,ax = plt.subplots(figsize=(16,10))
            sns.lineplot(x='TRADE_DT',y='B_ANAL_NET_CNBD',data=df,ax=ax,palette='Blues_d')
            plt.ylabel('估价净价')
            plt.xlabel('交易日期')
            plt.title('债券估价净价变动趋势:{0} {2}({6},{7}) \n 债券违约日期:{1},发行评级:{8},计息起始日:{3},兑付日:{4},债券类别:{5}'
                    .format(windcode,bond_default_date,entname,B_INFO_CARRYDATE,B_INFO_PAYMENTDATE,
                            bond_type,S_INFO_COMPIND_NAME1,B_AGENCY_GUARANTORNATURE,BOND_RATING))

            # plt.axvline(x=bond_default_date,c='red',ls='-')
            # plt.text(x=bond_default_date,y=df['B_ANAL_NET_CNBD'].median(),s='首次违约日期')
            # plt.axvline(x=data_dt,c='green',ls=':')
            # plt.text(x=data_dt,y=df['B_ANAL_NET_CNBD'].median(),s='观察日期')
            if file_path:
                plt.savefig(os.path.join(file_path,r'{}_{}.jpg'.format(entname,windcode)))
            return plt.show()
        else:
            print('无数据!!!')

    def get_group_data_dt_v2(self,default_type='v2'):    
        """获取各个时点的产业债好坏客户样本，坏客户表现期为半年"""
        ent_base_info = self.get_entid_baseinfo() # 主体基本信息
        if default_type == 'v1':
            ent_default_df =  self.get_default_all() # 主体历史违约信息
        if default_type == 'v2':
            ent_default_df =  self.get_default_all_v2() # 主体历史违约信息-WIND终端
        security_entid = self.get_security_entid() # 主体各时点存续信息
        ent_base_info_default = pd.merge(ent_base_info,ent_default_df,on=['S_INFO_COMPCODE','S_INFO_COMPNAME'],how='left')

        data_20180630 = pd.merge(ent_base_info_default,security_entid[security_entid['data_dt']=='20180630'],on=['S_INFO_COMPCODE','S_INFO_COMPNAME'],how='inner')
        data_20190101 = pd.merge(ent_base_info_default,security_entid[security_entid['data_dt']=='20190101'],on=['S_INFO_COMPCODE','S_INFO_COMPNAME'],how='inner')
        data_20190630 = pd.merge(ent_base_info_default,security_entid[security_entid['data_dt']=='20190630'],on=['S_INFO_COMPCODE','S_INFO_COMPNAME'],how='inner')
        data_20200101 = pd.merge(ent_base_info_default,security_entid[security_entid['data_dt']=='20200101'],on=['S_INFO_COMPCODE','S_INFO_COMPNAME'],how='inner')
        data_20200630 = pd.merge(ent_base_info_default,security_entid[security_entid['data_dt']=='20200630'],on=['S_INFO_COMPCODE','S_INFO_COMPNAME'],how='inner')
        data_20210101 = pd.merge(ent_base_info_default,security_entid[security_entid['data_dt']=='20210101'],on=['S_INFO_COMPCODE','S_INFO_COMPNAME'],how='inner')
        data_20210630 = pd.merge(ent_base_info_default,security_entid[security_entid['data_dt']=='20210630'],on=['S_INFO_COMPCODE','S_INFO_COMPNAME'],how='inner')
        data_all = pd.concat([data_20180630,data_20190101,data_20190630,data_20200101,data_20200630,data_20210101,data_20210630]) # 各时点存续主体汇总

        ## 非违约主体
        df_good = data_all[data_all['EntDefaultFirstDate'].isna()]

        ## 违约主体
        df_bad = data_all[data_all['EntDefaultFirstDate'].notna()]

        df_bad_20180630 = df_bad[df_bad['data_dt']=='20180630']
        df_bad_20180630['EntDefaultFirstDate'] =  df_bad_20180630['EntDefaultFirstDate'].astype(int)
        mark = (df_bad_20180630['EntDefaultFirstDate']>20180630) & (df_bad_20180630['EntDefaultFirstDate']<=20190101)
        df_bad_20180630 = df_bad_20180630[mark]

        df_bad_20190101 = df_bad[df_bad['data_dt']=='20190101']
        df_bad_20190101['EntDefaultFirstDate'] =  df_bad_20190101['EntDefaultFirstDate'].astype(int)
        mark = (df_bad_20190101['EntDefaultFirstDate']>20190101) & (df_bad_20190101['EntDefaultFirstDate']<=20190630)
        df_bad_20190101 = df_bad_20190101[mark]

        df_bad_20190630 = df_bad[df_bad['data_dt']=='20190630']
        df_bad_20190630['EntDefaultFirstDate'] =  df_bad_20190630['EntDefaultFirstDate'].astype(int)
        mark = (df_bad_20190630['EntDefaultFirstDate']>20190630) & (df_bad_20190630['EntDefaultFirstDate']<=20200101)
        df_bad_20190630 = df_bad_20190630[mark]

        df_bad_20200101 = df_bad[df_bad['data_dt']=='20200101']
        df_bad_20200101['EntDefaultFirstDate'] =  df_bad_20200101['EntDefaultFirstDate'].astype(int)
        mark = (df_bad_20200101['EntDefaultFirstDate']>20200101) & (df_bad_20200101['EntDefaultFirstDate']<=20200630)
        df_bad_20200101 = df_bad_20200101[mark]

        df_bad_20200630 = df_bad[df_bad['data_dt']=='20200630']
        df_bad_20200630['EntDefaultFirstDate'] =  df_bad_20200630['EntDefaultFirstDate'].astype(int)
        mark = (df_bad_20200630['EntDefaultFirstDate']>20200630) & (df_bad_20200630['EntDefaultFirstDate']<=20210101)
        df_bad_20200630 = df_bad_20200630[mark]

        df_bad_20210101 = df_bad[df_bad['data_dt']=='20210101']
        df_bad_20210101['EntDefaultFirstDate'] =  df_bad_20210101['EntDefaultFirstDate'].astype(int)
        mark = (df_bad_20210101['EntDefaultFirstDate']>20210101) & (df_bad_20210101['EntDefaultFirstDate']<=20210630)
        df_bad_20210101 = df_bad_20210101[mark]

        df_bad_20210630 = df_bad[df_bad['data_dt']=='20210630']
        df_bad_20210630['EntDefaultFirstDate'] =  df_bad_20210630['EntDefaultFirstDate'].astype(int)
        mark = (df_bad_20210630['EntDefaultFirstDate']>20210630) & (df_bad_20210630['EntDefaultFirstDate']<=20220101)
        df_bad_20210630 = df_bad_20210630[mark]

        df_bad_all = pd.concat([df_bad_20180630,df_bad_20190101,df_bad_20190630,df_bad_20200101,df_bad_20200630,df_bad_20210101,df_bad_20210630])

        DataAll = pd.concat([df_good,df_bad_all])
        DataAll['is_default'] = (DataAll['EntDefaultFirstDate'].notna()).astype(int)

        print(DataAll.groupby(['data_dt'])['S_INFO_COMPCODE'].nunique())
        print(DataAll.groupby(['data_dt'])['is_default'].sum())
        
        return DataAll
    
    def get_group_data_dt_v1(self,default_type='v2'):    
        """获取各个时点的产业债好坏客户样本，坏客户表现期为2年"""
        ent_base_info =  self.get_entid_baseinfo() # 主体基本信息
        if default_type == 'v1':
            ent_default_df =  self.get_default_all() # 主体历史违约信息
        if default_type == 'v2':
            ent_default_df =  self.get_default_all_v2() # 主体历史违约信息-WIND终端        
        security_entid =  self.get_security_entid() # 主体各时点存续信息
        ent_base_info_default = pd.merge(ent_base_info,ent_default_df,on=['S_INFO_COMPCODE','S_INFO_COMPNAME'],how='left')

        data_20180630 = pd.merge(ent_base_info_default,security_entid[security_entid['data_dt']=='20180630'],on=['S_INFO_COMPCODE','S_INFO_COMPNAME'],how='inner')
        data_20200630 = pd.merge(ent_base_info_default,security_entid[security_entid['data_dt']=='20200630'],on=['S_INFO_COMPCODE','S_INFO_COMPNAME'],how='inner')
        data_all = pd.concat([data_20180630,data_20200630]) # 各时点存续主体汇总

        ## 非违约主体
        df_good = data_all[data_all['EntDefaultFirstDate'].isna()]

        ## 违约主体
        df_bad = data_all[data_all['EntDefaultFirstDate'].notna()]

        df_bad_20180630 = df_bad[df_bad['data_dt']=='20180630']
        df_bad_20180630['EntDefaultFirstDate'] =  df_bad_20180630['EntDefaultFirstDate'].astype(int)
        mark = (df_bad_20180630['EntDefaultFirstDate']>20180630) & (df_bad_20180630['EntDefaultFirstDate']<=20200630)
        df_bad_20180630 = df_bad_20180630[mark]


        df_bad_20200630 = df_bad[df_bad['data_dt']=='20200630']
        df_bad_20200630['EntDefaultFirstDate'] =  df_bad_20200630['EntDefaultFirstDate'].astype(int)
        mark = (df_bad_20200630['EntDefaultFirstDate']>20200630) & (df_bad_20200630['EntDefaultFirstDate']<=20211231)
        df_bad_20200630 = df_bad_20200630[mark]

        df_bad_all = pd.concat([df_bad_20180630,df_bad_20200630])

        DataAll = pd.concat([df_good,df_bad_all])
        DataAll['is_default'] = (DataAll['EntDefaultFirstDate'].notna()).astype(int)

        print(DataAll.groupby(['data_dt'])['S_INFO_COMPCODE'].nunique())
        print(DataAll.groupby(['data_dt'])['is_default'].sum())
        
        return DataAll

    def get_group_data_dt_v3(self,default_type='v2'):    
        """获取各个时点的产业债好坏客户样本，坏客户表现期为2年"""
        ent_base_info =  self.get_entid_baseinfo() # 主体基本信息
        if default_type == 'v1':
            ent_default_df =  self.get_default_all() # 主体历史违约信息
        if default_type == 'v2':
            ent_default_df =  self.get_default_all_v2() # 主体历史违约信息-WIND终端        
        security_entid =  self.get_security_entid() # 主体各时点存续信息
        ent_base_info_default = pd.merge(ent_base_info,ent_default_df,on=['S_INFO_COMPCODE','S_INFO_COMPNAME'],how='left')

        data_20190630 = pd.merge(ent_base_info_default,security_entid[security_entid['data_dt']=='20190630'],on=['S_INFO_COMPCODE','S_INFO_COMPNAME'],how='inner')
        data_20210101 = pd.merge(ent_base_info_default,security_entid[security_entid['data_dt']=='20210101'],on=['S_INFO_COMPCODE','S_INFO_COMPNAME'],how='inner')
        data_all = pd.concat([data_20190630,data_20210101]) # 各时点存续主体汇总

        ## 非违约主体
        df_good = data_all[data_all['EntDefaultFirstDate'].isna()]

        ## 违约主体
        df_bad = data_all[data_all['EntDefaultFirstDate'].notna()]

        df_bad_20190630 = df_bad[df_bad['data_dt']=='20190630']
        df_bad_20190630['EntDefaultFirstDate'] =  df_bad_20190630['EntDefaultFirstDate'].astype(int)
        mark = (df_bad_20190630['EntDefaultFirstDate']>20190630) & (df_bad_20190630['EntDefaultFirstDate']<=20210630)
        df_bad_20190630 = df_bad_20190630[mark]


        df_bad_20210101 = df_bad[df_bad['data_dt']=='20210101']
        df_bad_20210101['EntDefaultFirstDate'] =  df_bad_20210101['EntDefaultFirstDate'].astype(int)
        mark = (df_bad_20210101['EntDefaultFirstDate']>20210101) & (df_bad_20210101['EntDefaultFirstDate']<=20211231)
        df_bad_20210101 = df_bad_20210101[mark]

        df_bad_all = pd.concat([df_bad_20190630,df_bad_20210101])

        DataAll = pd.concat([df_good,df_bad_all])
        DataAll['is_default'] = (DataAll['EntDefaultFirstDate'].notna()).astype(int)

        print(DataAll.groupby(['data_dt'])['S_INFO_COMPCODE'].nunique())
        print(DataAll.groupby(['data_dt'])['is_default'].sum())
        
        return DataAll


    def get_group_data_dt_v4(self,default_type='v2'):    
        """获取各个时点的产业债好坏客户样本，坏客户表现期为1年"""
        ent_base_info = self.get_entid_baseinfo() # 主体基本信息
        if default_type == 'v1':
            ent_default_df =  self.get_default_all() # 主体历史违约信息
        if default_type == 'v2':
            ent_default_df =  self.get_default_all_v2() # 主体历史违约信息-WIND终端

        security_entid = self.get_security_entid() # 主体各时点存续信息
        ent_base_info_default = pd.merge(ent_base_info,ent_default_df,on=['S_INFO_COMPCODE','S_INFO_COMPNAME'],how='left')

        data_20180630 = pd.merge(ent_base_info_default,security_entid[security_entid['data_dt']=='20180630'],on=['S_INFO_COMPCODE','S_INFO_COMPNAME'],how='inner')
        data_20190630 = pd.merge(ent_base_info_default,security_entid[security_entid['data_dt']=='20190630'],on=['S_INFO_COMPCODE','S_INFO_COMPNAME'],how='inner')
        data_20200630 = pd.merge(ent_base_info_default,security_entid[security_entid['data_dt']=='20200630'],on=['S_INFO_COMPCODE','S_INFO_COMPNAME'],how='inner')
        data_all = pd.concat([data_20180630,data_20190630,data_20200630]) # 各时点存续主体汇总

        ## 非违约主体
        df_good = data_all[data_all['EntDefaultFirstDate'].isna()]

        ## 违约主体
        df_bad = data_all[data_all['EntDefaultFirstDate'].notna()]

        df_bad_20180630 = df_bad[df_bad['data_dt']=='20180630']
        df_bad_20180630['EntDefaultFirstDate'] =  df_bad_20180630['EntDefaultFirstDate'].astype(int)
        mark = (df_bad_20180630['EntDefaultFirstDate']>20180630) & (df_bad_20180630['EntDefaultFirstDate']<=20190630)
        df_bad_20180630 = df_bad_20180630[mark]


        df_bad_20190630 = df_bad[df_bad['data_dt']=='20190630']
        df_bad_20190630['EntDefaultFirstDate'] =  df_bad_20190630['EntDefaultFirstDate'].astype(int)
        mark = (df_bad_20190630['EntDefaultFirstDate']>20190630) & (df_bad_20190630['EntDefaultFirstDate']<=20200630)
        df_bad_20190630 = df_bad_20190630[mark]


        df_bad_20200630 = df_bad[df_bad['data_dt']=='20200630']
        df_bad_20200630['EntDefaultFirstDate'] =  df_bad_20200630['EntDefaultFirstDate'].astype(int)
        mark = (df_bad_20200630['EntDefaultFirstDate']>20200630) & (df_bad_20200630['EntDefaultFirstDate']<=20210630)
        df_bad_20200630 = df_bad_20200630[mark]

        df_bad_all = pd.concat([df_bad_20180630,df_bad_20190630,df_bad_20200630])

        DataAll = pd.concat([df_good,df_bad_all])
        DataAll['is_default'] = (DataAll['EntDefaultFirstDate'].notna()).astype(int)

        print(DataAll.groupby(['data_dt'])['S_INFO_COMPCODE'].nunique())
        print(DataAll.groupby(['data_dt'])['is_default'].sum())
        
        return DataAll

    def get_event_features(self,data_dt):
        event_df = self.get_EventInfoALL(data_dt)
        event_type_all = event_df['EVTTYPE1'].unique().tolist()
        dat = pd.DataFrame(event_df['S_INFO_COMPCODE'].unique(),columns = ['S_INFO_COMPCODE'])
        for one in event_type_all:
            emp = event_df[event_df['EVTTYPE1']==one]
            if len(emp) == 0:
                emp_group = pd.DataFrame({'S_INFO_COMPCODE':event_df['S_INFO_COMPCODE'].unique(),'event_type_'+one:0})
            else:
                emp_group = emp.groupby(['S_INFO_COMPCODE'])['EVTTYPE1'].count().reset_index().rename(columns={'EVTTYPE1':'event_type_'+one})

            dat = pd.merge(dat,emp_group,on=['S_INFO_COMPCODE'],how='left')
            dat = dat.fillna(0)
        return event_type_all,dat

    def PlotEventCntDistr(self,data_dt,file_path):
        # base_info_target = self.get_group_data_dt_v1()
        base_info_target = self.get_group_data_dt_v3() #2019630,20210630 -wind终端违约
        base_info_target = base_info_target[base_info_target['data_dt']=='20190630']
        event_type_all,event_features = self.get_event_features(data_dt)
        event_columns = event_features.columns[1:]

        # 合并
        event_features_target = pd.merge(base_info_target,event_features,on=['S_INFO_COMPCODE'],how='left')
        for col in event_features.columns[1:]:
            event_features_target[col] = event_features_target[col].fillna(0)

        df1 = event_features_target[event_features_target['data_dt']==data_dt]
        total_ratio = df1['is_default'].sum()/len(df1)

        df1_bins = df1.copy()
        bins = [-np.inf,0,1,3,5,10,np.inf]
        for col in event_features.columns[1:]:
            df1_bins[col] = pd.cut(df1[col],bins)
        
        iv_dt = iv_df(df1_bins,df1_bins.columns[7:],in_dict={'target':'is_default'}).rename(columns={'IV':'train_iv'})
        iv_dict=  {}
        for vars,row in iv_dt.iterrows():
            iv_dict[row['指标英文名称']] = row['train_iv']
        
        for col in event_features.columns[1:]:
            dat1 = df1_bins.groupby([col])['is_default'].agg({'total':'count','bad':'sum'}).reset_index()
            dat1['bad_ratio'] = dat1['bad']/dat1['total']
            fig,ax = plt.subplots(figsize=(16,10))
            sns.barplot(x=col,y='bad_ratio',data=dat1,ax=ax,palette='Blues_d')
            plt.ylabel('违约占比')
            plt.xlabel("近2年事件发生次数")
            plt.title('{}(IV:{}):分箱违约占比分布(样本总体违约占比:{:.2%})'.format(col,round(iv_dict[col],4),total_ratio))
            # plt.title('{}:分箱违约占比分布(样本总体违约占比:{:.2%})'.format(col,total_ratio))
            xticks = ax.get_xticks()
            for i in range(len(dat1)):
                xy = (xticks[i],dat1['bad_ratio'][i]*1.02)
                s = '{:.2%}({})'.format(round(dat1['bad_ratio'][i],6),dat1['bad'][i].sum())
                ax.annotate(
                    s = s,
                    xy = xy,
                    fontsize = 12,
                    color = "blue",
                    ha = "center",
                    va = "baseline"
                )
            plt.savefig(os.path.join(file_path,r'{}.jpg'.format(col.replace('/',''))))
        return df1,df1_bins

    def get_BondDefaultEventTrendByWindcode(self,windcode,file_path):
        """债券违约事件序列"""
        windcode_info = self.get_windcode_baseinfo()
        bond_info = windcode_info[windcode_info['WINDCODE_TYPE']=='bond']
        bond_info['B_INFO_CARRYDATE'] = bond_info['B_INFO_CARRYDATE'].astype(int)
        
        sql  = """select B_INFO_WINDCODE,B_DEFAULT_DATE,B_DEFAULT_CODE
            from CBondDefaultReportform"""
        data = read_mysql(sql)
        data['B_DEFAULT_CODE'] = data['B_DEFAULT_CODE'].map(lambda x:str(x).split('.')[0])
        data['B_DEFAULT_DATE'] = data['B_DEFAULT_DATE'].astype(int)

        B_DEFAULT_CODE_dict = {'204020021':'担保违约',
                                '204055008':'触发交叉违约',
                                '204055012':'未按时兑付本金',
                                '204055013':'未按时兑付利息',
                                '204055014':'未按时兑付本息',
                                '204055015':'未按时兑付回售款',
                                '204055016':'未按时兑付回售款和利息',
                                '204055017':'提前到期未兑付',
                                '204055018':'技术性违约',
                                '204055030':'本息展期'}


        data['B_DEFAULT_CODE'] = data['B_DEFAULT_CODE'].map(B_DEFAULT_CODE_dict)

        data = data[data['B_INFO_WINDCODE'].isin(bond_info[bond_info['B_INFO_CARRYDATE']>20180630]['S_INFO_WINDCODE'].unique().tolist())]
        data = data.sort_values(by=['B_INFO_WINDCODE','B_DEFAULT_DATE'],ascending=True)
        df = data[data['B_INFO_WINDCODE']==windcode]
        print(df)
        df['id'] = np.arange(1,len(df)+1,1)
        df['B_DEFAULT_DATE'] = df['B_DEFAULT_DATE'].map(lambda x:str(x)[0:4] + '-' + str(x)[4:6] + '-' + str(x)[6:8])
        df['B_DEFAULT_DATE'] = pd.to_datetime(df['B_DEFAULT_DATE'],format = "%Y-%m-%d")
        df = df.reset_index(drop=True)
        entid = self.get_EntidByBond(windcode)
        entname = self.get_EntnameByEntid(entid) # 企业名称

        bond_info_dict = self.get_BondInfoByWindcode(windcode)
        B_INFO_CARRYDATE = bond_info_dict['B_INFO_CARRYDATE']  # 债券计息起始日
        B_INFO_PAYMENTDATE = bond_info_dict['B_INFO_PAYMENTDATE'] # 债券计息兑付日
        B_ISSUE_FIRSTISSUE = bond_info_dict['B_ISSUE_FIRSTISSUE']  # 发行起始日
        BOND_RATING = bond_info_dict['BOND_RATING']  # 发行信用评级

        bond_type = self.get_BondType(windcode) # 债券类别
        S_INFO_COMPIND_NAME1,B_AGENCY_GUARANTORNATURE = self.get_BondEntIndustryProperty(entid) # 发债企业WIND行业和企业属性
        
        fig,ax = plt.subplots(figsize=(16,10))
        sns.lineplot(x='B_DEFAULT_DATE',y='id',data=df,ax=ax,palette='Blues_d')

        for index,row in df.iterrows():
            plt.text(x=row['B_DEFAULT_DATE'],y=row['id'],s=row['B_DEFAULT_CODE'])
            plt.ylabel('违约事件序号')
            plt.xlabel('违约事件发生日期')
            plt.title('违约事件序列:{0} {1}({5},{6}) \n 发行评级:{7},计息起始日:{2},兑付日:{3},债券类别:{4}'
                    .format(windcode,entname,B_INFO_CARRYDATE,B_INFO_PAYMENTDATE,
                            bond_type,S_INFO_COMPIND_NAME1,B_AGENCY_GUARANTORNATURE,BOND_RATING))

        if file_path:
            plt.savefig(os.path.join(file_path,r'{}.jpg'.format(entname)))
        return data

ibdp = IndBondDataPrepare()



#*************************数据预处理**********************************************

def create_dict(dat,useless_vars = []):
    """初始化EDA"""
    
    if useless_vars:
        X_col = list(set(dat.columns.tolist()).difference(set(useless_vars)))
    else:
        X_col = list(set(dat.columns.tolist()))

    df = dat[X_col].copy()
    vars_info,vars_dict = IndBondDataPrepare().get_vars_info()
    vars_info = vars_info[['指标一级分类', '指标二级分类', '指标英文名称', '指标中文名称']]
    vars_info = vars_info[vars_info['指标英文名称'].isin(df.columns)]
    vars_info = pd.merge(vars_info,df.dtypes.reset_index().rename(columns={'index':'指标英文名称',0:'数据类型'}),how='left',on='指标英文名称')
    vars_info['数据类型'] = vars_info['数据类型'].apply(
        lambda x: 'varchar' if x == 'object' else 'float' if x == 'float64' else 'int')
    return vars_info

def eda(X, var_dict, useless_vars=[]):
    """EDA数据统计"""
#     X = X[list(set(X.columns)-set(useless_vars))]
    var_dict = var_dict.set_index('指标英文名称')
    variable_summary = X.count().to_frame('非空个数')
    variable_summary.loc[:, '总个数'] = len(X)
    variable_summary.loc[:, '缺失值个数'] = X.isna().sum()
    variable_summary.loc[:, '缺失率'] = np.round(variable_summary['缺失值个数']/variable_summary['总个数'],3)
    variable_summary.loc[:, '唯一值个数'] = X.nunique()


    try:
        numerical_vars = var_dict.loc[(var_dict['数据类型'] != 'varchar') & ~pd.isnull(var_dict['指标英文名称']), '指标英文名称'].tolist()
        categorical_vars = var_dict.loc[(var_dict['数据类型'] == 'varchar') & ~pd.isnull(var_dict['指标英文名称']), '指标英文名称'].tolist()
        numerical_vars = list(set(numerical_vars).intersection(set(X.columns)))
        categorical_vars = list(set(categorical_vars).intersection(set(X.columns)))
    except:
        numerical_vars = X.dtypes[X.dtypes!='object'].index.values
        categorical_vars = X.dtypes[X.dtypes=='object'].index.values

    if len(numerical_vars) > 0:
        X_numerical = X[numerical_vars].apply(lambda x: x.astype(float), 0)
        numerical_vars_summary = X_numerical.mean().round(6).to_frame('mean')
        numerical_vars_summary.loc[:, 'std'] = X_numerical.std().round(6)
        numerical_vars_summary.loc[:, 'median'] = X_numerical.median().round(6)
        numerical_vars_summary.loc[:, 'min'] = X_numerical.min()
        numerical_vars_summary.loc[:, 'max'] = X_numerical.max()
        numerical_vars_summary.loc[:, 'p01'] = X_numerical.quantile(0.01)
        numerical_vars_summary.loc[:, 'p05'] = X_numerical.quantile(q=0.05)
        numerical_vars_summary.loc[:, 'p10'] = X_numerical.quantile(q=0.10)
        numerical_vars_summary.loc[:, 'p25'] = X_numerical.quantile(q=0.25)
        numerical_vars_summary.loc[:, 'p75'] = X_numerical.quantile(q=0.75)
        numerical_vars_summary.loc[:, 'p90'] = X_numerical.quantile(q=0.90)
        numerical_vars_summary.loc[:, 'p95'] = X_numerical.quantile(q=0.95)
        numerical_vars_summary.loc[:, 'p99'] = X_numerical.quantile(q=0.99)

        if len(numerical_vars) > 0:
            X_numerical = X[numerical_vars].apply(lambda x: x.astype(float), 0)
            numerical_vars_summary = X_numerical.mean().round(6).to_frame('mean')
            numerical_vars_summary.loc[:, 'std'] = X_numerical.std().round(6)
            numerical_vars_summary.loc[:, 'median'] = X_numerical.median().round(6)
            numerical_vars_summary.loc[:, 'min'] = X_numerical.min()
            numerical_vars_summary.loc[:, 'max'] = X_numerical.max()
            numerical_vars_summary.loc[:, 'p01'] = X_numerical.quantile(0.01)
            numerical_vars_summary.loc[:, 'p05'] = X_numerical.quantile(q=0.05)
            numerical_vars_summary.loc[:, 'p10'] = X_numerical.quantile(q=0.10)
            numerical_vars_summary.loc[:, 'p25'] = X_numerical.quantile(q=0.25)
            numerical_vars_summary.loc[:, 'p75'] = X_numerical.quantile(q=0.75)
            numerical_vars_summary.loc[:, 'p90'] = X_numerical.quantile(q=0.90)
            numerical_vars_summary.loc[:, 'p95'] = X_numerical.quantile(q=0.95)
            numerical_vars_summary.loc[:, 'p99'] = X_numerical.quantile(q=0.99)
        # the following are for categorical_vars
        if len(categorical_vars) > 0:
            X_categorical = X[categorical_vars].copy()
            categorical_vars_summary = X_categorical.nunique().to_frame('N_categories')
            cat_list = []
            for col in categorical_vars:
                if X_categorical[col].count() == 0:
                    pass
                else:
                    cat_count = X_categorical[col].value_counts().head(3)
                    if len(cat_count) == 3:
                        col_result = pd.Series({'1st类别': str(cat_count.index.values[0]) + ' #=' + str(cat_count.iloc[0])\
                                                         + ', %=' + str(np.round(cat_count.iloc[0] * 1.0 / len(X), 2)),\
                                                '2nd类别': str(cat_count.index.values[1]) + ' #=' + str(cat_count.iloc[1])\
                                                         + ', %=' + str(np.round(cat_count.iloc[1] * 1.0 / len(X), 2)),\
                                                '3rd类别': str(cat_count.index.values[2]) + ' #=' + str(cat_count.iloc[2])\
                                                         + ', %=' + str(np.round(cat_count.iloc[2] * 1.0 / len(X), 2))\
                                                })\
                                                .to_frame().transpose()
                    elif len(cat_count) == 2:
                        col_result = pd.Series({'1st类别': str(cat_count.index.values[0]) + ' #=' + str(cat_count.iloc[0])\
                                                         + ', %=' + str(np.round(cat_count.iloc[0] * 1.0 / len(X), 2)),\
                                                '2nd类别': str(cat_count.index.values[1]) + ' #=' + str(cat_count.iloc[1])\
                                                         + ', %=' + str(np.round(cat_count.iloc[1] * 1.0 / len(X), 2)),\
                                                })\
                                                .to_frame().transpose()
                    elif len(cat_count) == 1:
                        col_result = pd.Series({'1st类别': str(cat_count.index.values[0]) + ' #=' + str(cat_count.iloc[0])\
                                                        + ', %=' + str(np.round(cat_count.iloc[0] * 1.0 / len(X), 2))})\
                                                .to_frame().transpose()
                    else:
                        pass

                    col_result.index = [col]
                    cat_list.append(col_result)

            cat_df = pd.concat(cat_list)

        # merge all summaries
        if len(numerical_vars) > 0 and len(categorical_vars) > 0:
            all_variable_summary = variable_summary.merge(numerical_vars_summary, how='left', left_index=True, right_index=True)
            all_variable_summary = all_variable_summary.merge(categorical_vars_summary, how='left', left_index=True, right_index=True)
            all_variable_summary = all_variable_summary.merge(cat_df, how='left', left_index=True, right_index=True)
        elif len(numerical_vars) > 0 and len(categorical_vars) == 0:
            all_variable_summary = variable_summary.merge(numerical_vars_summary, how='left', left_index=True, right_index=True)
        elif len(categorical_vars) > 0 and len(numerical_vars) == 0:
            all_variable_summary = variable_summary.merge(categorical_vars_summary, how='left', left_index=True, right_index=True)
            all_variable_summary = all_variable_summary.merge(cat_df, how='left', left_index=True, right_index=True)
        
        res = pd.merge(var_dict,all_variable_summary,how='left',right_index=True,left_index=True)
    return res.reset_index()

def process_missing(X,var_dict):
    """缺失值填充，保存每个特征的填充值字典"""

    if '指标一级分类' not in var_dict.columns and '指标英文名称' not in var_dict.columns:
        raise 'Check var_dict column names'
    X = X.replace([np.inf,-np.inf,'nan','None','NaN','null','Null','NULL',''],np.nan)
    
    unq_data_sources = var_dict['指标一级分类'].unique()
    var_fill_value = {}
    for data_source in unq_data_sources:
        data_sources_vars = var_dict.loc[var_dict['指标一级分类']==data_source, '指标英文名称'].unique()
        data_sources_vars = list(set(data_sources_vars).intersection(set(X.columns)))
        for col in data_sources_vars: 
            if col in X.columns:
                if X[col].isna().sum() == len(X):
                    fill_value = 0
                    X[col] = X[col].fillna(fill_value)
                if data_source != '舆情信息' and data_source != '关联风险':
                    fill_value = X[col].median()
                    # fill_value = -999
                    var_fill_value[col] = fill_value
                    X[col] = X[col].fillna(fill_value)
                elif data_source == '舆情信息' or data_source == '关联风险':
                    fill_value = 0
                    var_fill_value[col] = fill_value
                    X[col] = X[col].fillna(fill_value)
            else:
                print('The {} is not in data'.format(col))
    return X,var_fill_value
    
def process_missing_assign(X,var_fill_value,cols):
    """指定缺失值替换"""
    X = X.replace([np.inf,-np.inf,'nan','None','NaN','null','Null','NULL',''],np.nan)
    for col in cols:
        if col in X.columns:
            if col not in var_fill_value:
                print('warnings:the {} not in var_fill_value!'.format(col))
            X[col] = X[col].fillna(var_fill_value[col])
        else:
            print('The {} is not in data'.format(col))
    return X

def data_preprocess(dat, in_dict, var_dict=None,is_save=True):
    '''
    输入原始数据，输出缺失填充好的数据、字典、eda结果
    '''
    target = in_dict['target']
    index_id = in_dict['index_id']
    
    df = dat.copy()
#     df.columns = df.columns.str.replace(':','_').str.replace('/','_').str.replace('\\','_')
    #检查Y变量，如果有缺失，缺填充为0
    df[target] = df[target].fillna(0).astype(int)
    if df[target].nunique()>2:
        print('Warning: target种类大于2，仅取target=0或target=1')
        df = df[df[target].isin([0,1])]
    df.set_index(index_id, inplace=True)
    

    # 生成字典
    x_col = list(set(list(df.columns)) - set(in_dict['extra_col']) - set([target]))
    x_all = df[x_col]
    if var_dict is None:
        var_dict = create_dict(x_all,useless_vars = [])
    
    #数据缺失处理
    extra_col = in_dict['extra_col']
    X_cleaned,var_fill_value = process_missing(X = x_all,var_dict=var_dict)
    X_cleaned_sorted = X_cleaned.reindex(x_all.index.tolist())
    data_cleaned = pd.concat([df[extra_col+[target]], X_cleaned_sorted], axis=1)
    data_cleaned = data_cleaned[df.columns.tolist()]
    
    # EDA
    eda_df = eda(X = X_cleaned_sorted,var_dict=var_dict, useless_vars = [])

    if is_save:
        save_data_to_pickle(var_fill_value,file_path = in_dict['RESULT_PATH'],file_name='var_fill_value.pkl')
    return data_cleaned, var_dict, eda_df,var_fill_value



##****************************特征评估********************************************##
def get_maxks_split_point(data, var, target, min_sample=0.05):
    """ 计算KS值
    Args:
        data: DataFrame，待计算卡方分箱最优切分点列表的数据集
        var: 待计算的连续型变量名称
        target: 待计算的目标列Y的名称
        min_sample: int，分箱的最小数据样本，也就是数据量至少达到多少才需要去分箱，一般作用在开头或者结尾处的分箱点
    Returns:
        ks_v: KS值，float
        BestSplit_Point: 返回本次迭代的最优划分点，float
        BestSplit_Position: 返回最优划分点的位置，最左边为0，最右边为1，float
    """
    if len(data) < min_sample:
        ks_v, BestSplit_Point, BestSplit_Position = 0, -9999, 0.0
    else:
        freq_df = pd.crosstab(index=data[var], columns=data[target])
        freq_array = freq_df.values
        if freq_array.shape[1] == 1: # 如果某一组只有一个枚举值，如0或1，则数组形状会有问题，跳出本次计算
            # tt = np.zeros(freq_array.shape).T
            # freq_array = np.insert(freq_array, 0, values=tt, axis=1)
            ks_v, BestSplit_Point, BestSplit_Position = 0, -99999, 0.0
        else:
            bincut = freq_df.index.values
            tmp = freq_array.cumsum(axis=0)/(np.ones(freq_array.shape) * freq_array.sum(axis=0).T)
            tmp_abs = abs(tmp.T[0] - tmp.T[1])
            ks_v = tmp_abs.max()
            BestSplit_Point = bincut[tmp_abs.tolist().index(ks_v)]
            BestSplit_Position = tmp_abs.tolist().index(ks_v)/max(len(bincut) - 1, 1)
        
    return ks_v, BestSplit_Point, BestSplit_Position

def get_bestks_bincut(data, var, target, leaf_stop_percent=0.05):
    """ 计算最优分箱切分点
    Args:
        data: DataFrame，拟操作的数据集
        var: String，拟分箱的连续型变量名称
        target: String，Y列名称
        leaf_stop_percent: 叶子节点占比，作为停止条件，默认5%
    
    Returns:
        best_bincut: 最优的切分点列表，List
    """
    min_sample = len(data) * leaf_stop_percent
    best_bincut = []
    
    def cutting_data(data, var, target, min_sample, best_bincut):
        ks, split_point, position = get_maxks_split_point(data, var, target, min_sample)
        
        if split_point != -99999:
            best_bincut.append(split_point)
        
        # 根据最优切分点切分数据集，并对切分后的数据集递归计算切分点，直到满足停止条件
        # print("本次分箱的值域范围为{0} ~ {1}".format(data[var].min(), data[var].max()))
        left = data[data[var] < split_point]
        right = data[data[var] > split_point]
        
        # 当切分后的数据集仍大于最小数据样本要求，则继续切分
        if len(left) >= min_sample and position not in [0.0, 1.0]:
            cutting_data(left, var, target, min_sample, best_bincut)
        else:
            pass
        if len(right) >= min_sample and position not in [0.0, 1.0]:
            cutting_data(right, var, target, min_sample, best_bincut)
        else:
            pass
        return best_bincut
    best_bincut = cutting_data(data, var, target, min_sample, best_bincut)
    
    # 把切分点补上头尾
    best_bincut.append(data[var].min())
    best_bincut.append(data[var].max())
    best_bincut_set = set(best_bincut)
    best_bincut = list(best_bincut_set)
    
    best_bincut.remove(data[var].min())
    best_bincut.append(data[var].min()-1)
    # 排序切分点
    best_bincut.sort()
    
    return best_bincut

def get_qcut_bincut(data,var,qlist=[0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.75,0.80,0.95]):
    """基于分位数获取分箱分割点list"""
    bins = list(set([round(data[var].quantile(q=i),6) for i in qlist]))
    bins.sort()
    return bins

def get_var_bins(df,cols,in_dict,method = 'bestks'):
    """计算每个特征的分箱分割点，并保存到本地 var_bins.pkl"""
    var_bins = {}
    for col in cols:
        if method == 'qcut':
            bins = get_qcut_bincut(df,col)
        elif method == 'bestks':
            bins = get_bestks_bincut(df, col, in_dict['target'], leaf_stop_percent=0.05)
        
        bins = [-np.inf] + bins + [np.inf]
        var_bins[col] = bins   
    save_data_to_pickle(var_bins,file_path = in_dict['RESULT_PATH'],file_name='var_bins_{}.pkl'.format(method))
    return var_bins

def binning(dat,cols,var_bins):
    """离散化：基于分位数分箱"""
    df = dat.copy()
    for col in cols:
        bins = var_bins[col]
        df[col] = pd.cut(df[col],bins=bins)
    return df

def iv(df, y, features_cols=None, positive='bad|1', order=True):
    
    dt = df.copy(deep=True)
    if isinstance(y, str):
        y = [y]
    dt = dt[y+features_cols]
    #分箱
    dt = dt.fillna(0)
    for col in features_cols:
        bins = get_qcut_bincut(df_train, var)
        if dt[col].nunique() <=10:
            dt[col] = dt[col]
        else:
            dt[col] = pd.qcut(dt[col],10,duplicates='drop')
    
    # info_value
    ivlist = pd.DataFrame({
        'variable': features_cols,
        'info_value': [iv_xy(dt[i], dt[y[0]]) for i in features_cols]
    }, columns=['variable', 'info_value'])
    # sorting iv
    if order: 
        ivlist = ivlist.sort_values(by='info_value', ascending=False)
    return ivlist

def iv_xy(x, y):
    # good bad func
    def goodbad(df):
        names = {'good': (df['y']==0).sum(),'bad': (df['y']==1).sum()}
        return pd.Series(names)
    # iv calculation
    iv_total = pd.DataFrame({'x':x.astype('str'),'y':y}) \
      .fillna('missing') \
      .groupby('x') \
      .apply(goodbad) \
      .replace(0, 0) \
      .assign(
        DistrBad = lambda x: x.bad/sum(x.bad),
        DistrGood = lambda x: x.good/sum(x.good)
      ) \
      .assign(iv = lambda x: (x.DistrBad-x.DistrGood)*np.log(x.DistrBad/x.DistrGood)) \
     .replace(np.inf,0).replace(-np.inf,0)\
      .iv.sum()
    # return iv
    return iv_total

def psi_df(train,test,cols,is_transform_bins=True):
    psi_dt = pd.DataFrame({
        '指标英文名称': cols,
        'PSI': [psi(train,test,col,show_plot=False,is_transform_bins=is_transform_bins) for col in cols]
    })
    return psi_dt.sort_values(by='PSI',ascending=False)    

def iv_df(df,cols,in_dict):
    iv_dt = pd.DataFrame({
        '指标英文名称': cols,
        'IV': [iv_xy(df[col],df[in_dict['target']]) for col in cols]
    })
    return iv_dt.sort_values(by='IV',ascending=False)  



def psi(df_train,df_test,var,show_plot=False,is_transform_bins=False):
    """计算训练集和测试集的PSI"""
    if not is_transform_bins:
        train_bins = get_qcut_bincut(df_train, var)
        df_train_bins = pd.cut(df_train[var],bins=train_bins).value_counts().sort_index().reset_index(name='训练集总个数').rename(columns={'index':'bins'})
        df_test_bins = pd.cut(df_test[var],bins=train_bins).value_counts().sort_index().reset_index(name='验证集总个数').rename(columns={'index':'bins'})    
    else:
        df_train_bins = df_train[var].value_counts().sort_index().reset_index(name='训练集总个数').rename(columns={'index':'bins'})
        df_test_bins = df_test[var].value_counts().sort_index().reset_index(name='验证集总个数').rename(columns={'index':'bins'})       
    df_train_bins['训练集各组个数占比'] = df_train_bins['训练集总个数']/df_train_bins['训练集总个数'].sum()
    df_test_bins['验证集各组个数占比'] = df_test_bins['验证集总个数']/df_test_bins['验证集总个数'].sum()
    psi_dt = pd.merge(df_train_bins,df_test_bins,on=['bins'],how='left')
    psi_dt['占比差值'] = psi_dt['训练集各组个数占比']-psi_dt['验证集各组个数占比']
    psi_dt['psi'] = (psi_dt['训练集各组个数占比']-psi_dt['验证集各组个数占比']) * np.log10(psi_dt['训练集各组个数占比']/psi_dt['验证集各组个数占比'])
    psi_dt['psi'] = psi_dt['psi'].map(lambda x:round(x,4))
    psi_dt = psi_dt.replace([np.inf,-np.inf],0)
    psi_dt_melt = pd.melt(psi_dt,id_vars='bins',value_vars=['训练集各组个数占比','验证集各组个数占比'],var_name='data_type',value_name='count')
    psi_dt_melt['data_type'] = psi_dt_melt['data_type'].str.replace('各组个数占比','')
    if show_plot:
        sns.barplot(data=psi_dt_melt,x='bins',y='count',hue='data_type')

    return round(psi_dt['psi'].sum(),8)

##****************************模型评估********************************************##

def ks_compute(proba_arr, target_arr):
    """计算ks"""
    from scipy.stats import ks_2samp
    get_ks = lambda proba_arr, target_arr: ks_2samp(proba_arr[target_arr == 1], \
                                           proba_arr[target_arr == 0]).statistic
    ks_value = get_ks(proba_arr, target_arr)
    return ks_value

def format_confusion_matrix(confusion_matrix, type_name, placeholder_length=12):
    """混淆矩阵格式化"""
    res = ['          ## 混淆矩阵 ##\n']

    tmp = ['实际 \ 预测'] + type_name
    for tn in tmp:
        fm = '%-' + str(placeholder_length) + 's'
        res.append(fm % tn)  # 不换行输出每一列表头
    res.append('\n')

    for i, cm in enumerate(confusion_matrix):
        fm = '%-' + str(placeholder_length) + 's'
        res.append(fm % tmp[i + 1])  # 不换行输出每一行表头

        for c in cm:
            fm = '%-' + str(placeholder_length) + 'd'
            res.append(fm % c)  # 不换行输出每一行元素
        res.append('\n')
    return ''.join(res)

def calc_metrics(y_ture,y_pred,y_pred_class,labels=['正常', '违约']):
    """模型评估指标"""
    cm = confusion_matrix(y_ture, y_pred_class)
    metrics = {}
    metrics['KS'] = round(ks_compute(y_pred,y_ture),6)
    metrics['Recall'] = round(recall_score(y_ture, y_pred_class),6)
    metrics['Precision'] = round(precision_score(y_ture, y_pred_class),6)
    metrics['F1'] = round(f1_score(y_ture, y_pred_class),6)
    metrics['AUC'] = round(roc_auc_score(y_ture, y_pred), 6)
    print(format_confusion_matrix(cm, labels))
    return metrics


def get_woe(df,cols,vars_dict):
    res = pd.DataFrame(columns=['指标英文名称', '指标中文名称', 'bins', 'total', 'bad', 'good', 'bad_ratio',
       'DistrGood', 'DistrBad', 'woe', 'iv'])
    for col in cols:
        dt = df.groupby([col])['is_default'].agg({'total':'count','bad':'sum'}).reset_index().rename(columns={col:'bins'})
        dt['good'] = dt['total'] - dt['bad']
        dt['bad_ratio'] = dt['bad']/dt['total']
        dt['DistrGood'] = dt['good']/dt['good'].sum()
        dt['DistrBad'] = dt['bad']/dt['bad'].sum()
        dt['指标英文名称'] = col 
        dt['指标中文名称'] = vars_dict[col] 
        dt = dt[['指标英文名称', '指标中文名称','bins', 'total', 'bad', 'good', 'bad_ratio', 'DistrGood', 'DistrBad']]
        dt = dt.assign(woe = lambda x: np.log(x.DistrBad/x.DistrGood))
        dt = dt.assign(iv = lambda x: (x.DistrBad-x.DistrGood)*np.log(x.DistrBad/x.DistrGood))
        res = pd.concat([res,dt])
    res = res.replace([-np.inf,np.inf],[0,0]).fillna(0)
    res['bins'] = res['bins'].astype(str)
    return res


def var_base_select(ead_df,in_dict):
    """特征筛选:缺失率，IV，PSI,相关性"""
    iv_thresold = in_dict['features_select_thresold']['iv_thresold']
    psi_thresold = in_dict['features_select_thresold']['psi_thresold']
    missing_thresold = in_dict['features_select_thresold']['missing_thresold']

    ead_select = ead_df.copy()
    woe_mark = ead_df['阈值'].notna()
    missing_mark = ead_select['缺失率']<missing_thresold
    iv_mark = (ead_select['train_iv']>iv_thresold)&(ead_select['test_iv']>iv_thresold)
    psi_mark = (ead_select['test_psi']<psi_thresold)
    # IV变化筛选
    iv_change_mark = np.abs((ead_select['test_iv'] - ead_select['train_iv'])/ead_select['test_iv'])<1
     
    # ead_select = ead_select[(woe_mark)&(missing_mark)&(iv_mark)&(psi_mark)&(iv_change_mark)]
    ead_select = ead_select[(missing_mark)&(iv_mark)&(psi_mark)]
    base_select = ead_select['指标英文名称'].tolist()
    # 筛选后的特征
    print('筛选后的特征数:{}'.format(len(base_select)))
    # WOE单调性删除的特征
    woe_delete_vars = ead_df[~woe_mark]['指标英文名称'].tolist()
    iv_delete_vars = ead_df[~iv_mark]['指标英文名称'].tolist()
    psi_delete_vars = ead_df[~psi_mark]['指标英文名称'].tolist()
    missing_delete_vars = ead_df[~missing_mark]['指标英文名称'].tolist()
    iv_change_delete_vars = ead_df[~iv_change_mark]['指标英文名称'].tolist()
    return base_select,woe_delete_vars,iv_delete_vars,psi_delete_vars,missing_delete_vars,iv_change_delete_vars

def var_corr_select(df,corr_thresold,ead_df,cols,in_dict):
    
    # 保存相关性矩阵
    corr_df = df[cols].corr()
    # corr_df.to_excel(os.path.join(in_dict['RESULT_PATH'],'corr.xlsx'),index=False)
    
    # IV DataFrame
    iv_df = ead_df[ead_df['指标英文名称'].isin(cols)]
    corr_thresold = in_dict['features_select_thresold']['corr_thresold']
    # IV排序
    high_IV = {k:v for k,v in zip(iv_df['指标英文名称'],iv_df['train_iv'])}
    high_IV_sorted = sorted(high_IV.items(),key=lambda x:x[1],reverse=True)

    #两两间的线性相关性检验
    #1，将候选变量按照IV进行降序排列
    #2，计算第i和第i+1的变量的线性相关系数
    #3，对于系数超过阈值的两个变量，剔除IV较低的一个
    deleted_index = []
    cnt_vars = len(high_IV_sorted)
    for i in range(cnt_vars):
        if i in deleted_index:
            continue
        x1 = high_IV_sorted[i][0]
        for j in range(cnt_vars):
            if i == j or j in deleted_index:
                continue
            y1 = high_IV_sorted[j][0]
#             roh = np.corrcoef(df[x1],df[y1])[0,1]
            roh = corr_df.loc[x1,y1]
            if abs(roh)>corr_thresold:
                x1_IV = high_IV_sorted[i][1]
                y1_IV = high_IV_sorted[j][1]
                if x1_IV > y1_IV:
                    deleted_index.append(j)
                else:
                    deleted_index.append(i)

    var_select_corr = [high_IV_sorted[i][0] for i in range(cnt_vars) if i not in deleted_index]
    # 筛选后的特征
    print('筛选后的特征数:{}'.format(len(var_select_corr)))
    # 删除的特征
    corr_delete_vars = list(set(cols)-set(var_select_corr))
    return var_select_corr,corr_delete_vars,corr_df

def var_imp_select(df,cols,in_dict):
    train = df.copy()
    X_train = train[cols]
    y_train = train[in_dict['target']]

    model = BalancedRandomForestClassifier(n_estimators = in_dict['random_param']['n_estimators']
                                        , max_depth = in_dict['random_param']['max_depth']
                                        , class_weight = in_dict['random_param']['class_weight']
                                        , random_state = in_dict['random_param']['random_state']
                                            )
    model.fit(X_train,y_train)

    ## model变量重要性
    best_importance = model.feature_importances_
    model_importance = pd.DataFrame({'variable':cols,'importance':best_importance},columns=["variable", 'importance'])
    model_importance = model_importance.sort_values(by=['importance'],ascending=False)
    imp_select = model_importance[model_importance['importance']>0]['variable'].tolist()
    print('筛选后的特征数:{}'.format(len(imp_select)))
    return imp_select,model_importance


def plot_model_ks(y_label,y_pred,ax):
    """
    y_label:测试集的y
    y_pred:对测试集预测后的概率
    
    return:KS曲线
    """
    pred_list = list(y_pred) 
    label_list = list(y_label)
    total_bad = sum(label_list)
    total_good = len(label_list)-total_bad 
    items = sorted(zip(pred_list,label_list),key=lambda x:x[0]) 
    step = (max(pred_list)-min(pred_list))/200 
    
    pred_bin=[]
    good_rate=[] 
    bad_rate=[] 
    ks_list = [] 
    for i in range(1,201): 
        idx = min(pred_list)+i*step 
        pred_bin.append(idx) 
        label_bin = [x[1] for x in items if x[0]<idx] 
        bad_num = sum(label_bin)
        good_num = len(label_bin)-bad_num  
        goodrate = good_num/total_good 
        badrate = bad_num/total_bad
        ks = abs(goodrate-badrate) 
        good_rate.append(goodrate)
        bad_rate.append(badrate)
        ks_list.append(ks)
    
#     fig = plt.figure(figsize=(8,6))
#     ax = fig.add_subplot(1,1,1)
    ax.plot(pred_bin,good_rate,color='green',label='good_rate')
    ax.plot(pred_bin,bad_rate,color='red',label='bad_rate')
    ax.plot(pred_bin,ks_list,color='blue',label='good-bad')
    ax.set_title('KS曲线:{:.3f}'.format(max(ks_list)))
    ax.legend(loc='best')
    
    return plt.show(ax)


def plot_metrics(df,data_type,recall_precision_dt,in_dict):
    df['pred_bins'] = pd.qcut(df['pred'],10,duplicates = 'drop')
    score_default_df =  df.groupby(['pred_bins'])[in_dict['target']].agg({'total':'count','bad':np.sum}).sort_index().reset_index()
    score_default_df['bad_ratio'] = score_default_df['bad']/score_default_df['total']

    fig = plt.figure(figsize=(16,10),dpi=300)
    fig.suptitle(data_type)
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    sns.lineplot(x='FPR',y='TPR',data=recall_precision_dt,ax=ax1,palette='Blues_d')
    ax1.plot([0, 1], [0, 1], 'r--')
    sns.lineplot(x='recall',y='precision',data=recall_precision_dt,ax=ax2,palette='Blues_d')
    sns.barplot(x='pred_bins',y='bad_ratio',data=score_default_df,ax=ax4,palette='Blues_d')
    ax4.set_xticklabels(score_default_df['pred_bins'],rotation=90) # 刻度旋转 
    ax1.set_title('ROC曲线')
    ax2.set_title('PR曲线')
    ax3.set_title('KS曲线')
    ax4.set_title('预测概率分箱违约率分布')
    plot_model_ks(df[in_dict['target']],df['pred'],ax3)
    figname = os.path.join(in_dict['RESULT_PATH'],'{}_评估曲线.jpg'.format(data_type))
    fig.savefig(figname)
    return figname

def get_pred_class_threshold(df,in_dict):
    # 确定阈值 召回率固定0.8
    pred_class_threshold = np.linspace(0,1,50)
    precision_list = []
    recall_list = []
    Specificity_list = []
    TPR_list = []
    FPR_list = []
    for threshold in pred_class_threshold:
        df['pred_class'] =df['pred'].map(lambda x:1 if x>threshold else 0)
        TP = df[(df['pred_class'] == 1)&(df[in_dict['target']] == 1)].shape[0]
        FP = df[(df['pred_class'] == 1)&(df[in_dict['target']] == 0)].shape[0]
        TN = df[(df['pred_class'] == 0)&(df[in_dict['target']] == 0)].shape[0]
        FN = df[(df['pred_class'] == 0)&(df[in_dict['target']] == 1)].shape[0]
        Precision = TP/(TP+FP+1)
        Recall = TP/(TP+FN)
        Specificity = TN/(FP+TN)
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        precision = round(Precision,6)
        recall = round(Recall,6)
        precision_list.append(precision)
        recall_list.append(recall)
        Specificity_list.append(Specificity)
        TPR_list.append(TPR)
        FPR_list.append(FPR)

    recall_precision_dt = pd.DataFrame({'threshold':pred_class_threshold,'precision':precision_list,'recall':recall_list,
                                      'Specificity':Specificity_list,'TPR':TPR_list,'FPR':FPR_list})
    best_pred_class_threshold = recall_precision_dt[recall_precision_dt['precision']>=0.20].sort_values(by=['precision'],ascending=True).reset_index(drop=True)['threshold'][0]
    return best_pred_class_threshold,recall_precision_dt

def model_train(features_select,df_train_clean,df_test_clean,in_dict):
    train_df = df_train_clean.copy()
    test_df = df_test_clean.copy()

    train_df['DataSetType'] = 'train'
    test_df['DataSetType'] = 'test'

    DataAll = pd.concat([train_df,test_df],axis=0)
    print(DataAll.shape)
    print('训练集坏客户占比:{:.2%}'.format(train_df[in_dict['target']].sum()/len(train_df)))
    print('测试集坏客户占比:{:.2%}'.format(test_df[in_dict['target']].sum()/len(test_df)))

    train = DataAll[DataAll['DataSetType']=='train']
    test = DataAll[DataAll['DataSetType']=='test']
    
    X_train = train[features_select]
    y_train = train[in_dict['target']]
    X_test = test[features_select]
    y_test = test[in_dict['target']]


    if in_dict['algorithm_type'] == 'BRF':
        model = BalancedRandomForestClassifier(n_estimators=500,max_depth=4,random_state=0)
        model = BalancedRandomForestClassifier(n_estimators=100,max_depth=3,random_state=0)
        model = BalancedRandomForestClassifier(n_estimators = in_dict['random_param']['n_estimators']
                                            , max_depth = in_dict['random_param']['max_depth']
                                            , class_weight = in_dict['random_param']['class_weight']
                                            , random_state = in_dict['random_param']['random_state']
                                                )
    elif in_dict['algorithm_type'] == 'RF':
        model = RandomForestClassifier(n_estimators = in_dict['random_param']['n_estimators']
                                            , max_depth = in_dict['random_param']['max_depth']
                                            , class_weight = in_dict['random_param']['class_weight']
                                            , random_state = in_dict['random_param']['random_state']
                                            , ccp_alpha = in_dict['random_param']['ccp_alpha']
                                            , min_samples_split = in_dict['random_param']['min_samples_split']
                                                )
    elif in_dict['algorithm_type'] == 'ETC':
        model = ExtraTreesClassifier(n_estimators = in_dict['random_param']['n_estimators']
                                            , max_depth = in_dict['random_param']['max_depth']
                                            , class_weight = in_dict['random_param']['class_weight']
                                            , random_state = in_dict['random_param']['random_state']
                                                )

    elif in_dict['algorithm_type'] == 'ABC':
        model = AdaBoostClassifier(n_estimators = in_dict['random_param']['n_estimators']
                                            , random_state = in_dict['random_param']['random_state']
                                                )

    elif in_dict['algorithm_type'] == 'GBC':
        model = GradientBoostingClassifier(n_estimators = in_dict['random_param']['n_estimators']
                                            , max_depth = in_dict['random_param']['max_depth']
                                            , random_state = in_dict['random_param']['random_state']
                                                )
    model.fit(X_train,y_train)

    y_pred_train =  model.predict_proba(X_train)[:,1]
    y_pred_test = model.predict_proba(X_test)[:,1]

    train['pred'] = y_pred_train
    test['pred'] = y_pred_test
    
    # 模型参数
    params_df = pd.DataFrame(pd.Series(model.get_params())).reset_index()
    params_df.columns = ['参数名称','参数值']

    ## model变量重要性
    best_importance = model.feature_importances_
    model_importance = pd.DataFrame(columns=["variable", 'importance'])
    model_importance['variable'] = features_select
    model_importance['importance'] = best_importance
    
    save_data_to_pickle(features_select, file_path=in_dict['RESULT_PATH'], file_name='features_select.pkl')
    save_data_to_pickle(model, file_path=in_dict['RESULT_PATH'], file_name='model.pkl')

    return model,train,test,params_df,model_importance
    
    
def model_mertics(train,test,in_dict):
    best_pred_class_threshold,train_recall_precision_dt = get_pred_class_threshold(train,in_dict)
    # 是否使用固定阈值
    # if 1 == 1:
    #     best_pred_class_threshold = 0.5 
    _,test_recall_precision_dt = get_pred_class_threshold(test,in_dict)
    
    # 阈值 
#     threshold = best_pred_class_threshold
    threshold = 0.3469
    train['pred_class'] =train['pred'].map(lambda x:1 if x>threshold else 0)
    test['pred_class'] =test['pred'].map(lambda x:1 if x>threshold else 0)
    
    train_metrics = calc_metrics(train[in_dict['target']],train['pred'],train['pred_class'])
    test_metrics = calc_metrics(test[in_dict['target']],test['pred'],test['pred_class'])

    train_metric_df = pd.DataFrame(pd.Series(train_metrics)).T
    train_metric_df['DataType'] = 'train'
    if len(in_dict['obser_datepoint']['train_good'])>1:
        train_metric_df['obser_datepoint'] = str(in_dict['obser_datepoint']['train_good'])
    else:
        train_metric_df['obser_datepoint'] = in_dict['obser_datepoint']['train_good']

    test_metric_df = pd.DataFrame(pd.Series(test_metrics)).T
    test_metric_df['DataType'] = 'test'
    if len(in_dict['obser_datepoint']['train_good'])>1:
        test_metric_df['obser_datepoint'] = str(in_dict['obser_datepoint']['train_good'])
    else:
        test_metric_df['obser_datepoint'] = in_dict['obser_datepoint']['train_good']


    metric_df = pd.concat([train_metric_df,test_metric_df])
    
    train_figname = plot_metrics(train,'train',train_recall_precision_dt,in_dict)
    test_figname = plot_metrics(test,'test',test_recall_precision_dt,in_dict)
    metircs_plot_dict = {'train':train_figname,'test':test_figname}
    return best_pred_class_threshold,metric_df,metircs_plot_dict,train_recall_precision_dt,test_recall_precision_dt

