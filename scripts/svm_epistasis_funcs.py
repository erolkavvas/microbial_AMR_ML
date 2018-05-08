import pandas as pd
import os, sys
from pickle import *
from tqdm import tqdm
import ast
from itertools import cycle
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
import scipy.stats as ss
import json

# sys.path.append('/home/user/notebook/erol/TB Project/Modules')
# from naked import *
# from model_troubleshooting import *
# import networkx as nx
# from mtb_functions import *

from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_selection import SelectFromModel, SelectPercentile, VarianceThreshold, chi2, f_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import svm

# Json related
def save_json_obj(sim_json, file_name):
    with open(file_name, 'w') as fp:
        json.dump(sim_json, fp)
        
def load_json_obj(file_name):
    with open(file_name, 'r') as fp:
        data = json.load(fp)
        return data

# -------------------------------------------------------------------------------------------------------
# ------------------------------------------ Machine Learning functions ---------------------------------
# -------------------------------------------------------------------------------------------------------
def remove_key_antibiotic_clusters(X, drug_name):
    drug_clusts = []
    print drug_name
    for col in X.columns:
        if ('Cluster551_' in col) and "ethambutol" not in drug_name:
            drug_clusts.append(col)
        elif ('Cluster4244_' in col) and "ethambutol" not in drug_name:
            drug_clusts.append(col)
        elif "Cluster491_" in col and "rifampicin" not in drug_name:
            drug_clusts.append(col)
        elif "Cluster415_" in col and "rifampicin" not in drug_name:
            drug_clusts.append(col)   
        elif "Cluster7435_" in col and "streptomycin" not in drug_name:
            drug_clusts.append(col)
        elif "Cluster5240_" in col and "streptomycin" not in drug_name:
            drug_clusts.append(col)
        elif "Cluster3930_" in col and "pyrazinamide" not in drug_name:
            drug_clusts.append(col)
        elif "Cluster1116_" in col and "isoniazid" not in drug_name:
            drug_clusts.append(col)
        elif "Cluster893_" in col and "ofloxacin" not in drug_name:
            drug_clusts.append(col)
        elif "Cluster1613_" in col and "pyrazinamide" not in drug_name: 
            drug_clusts.append(col)
        elif "Cluster2122_" in col and "ethionamide" not in drug_name: 
            drug_clusts.append(col)
        elif "Cluster4486" in col and "4-aminosalicylic_acid" not in drug_name: # Thymidilate synthase
            drug_clusts.append(col)
        
    print "removing... ", len(drug_clusts) , "clusters"
    X.drop(drug_clusts, axis=1, inplace=True)
    return X


def drug_type_binary(x):
    if x == "S":
        return 0
    elif x == "R":
        return 1

def get_target_data(gene_vs_genome_frame, resistance_data, drug_choice):
    '''Takes in all dataframes and returns them in a usable form.
        Also, the training set is balanced for susceptible and resistance strains'''
    all_strain_data = list(gene_vs_genome_frame.index)
    drug_strain_subset = resistance_data.ix[all_strain_data]
    suscept_samples = list(drug_strain_subset[drug_strain_subset.loc[:, drug_choice] == "S"].index)
    resist_samples= list(drug_strain_subset[drug_strain_subset.loc[:, drug_choice] == "R"].index)
    list_of_random_items = suscept_samples + resist_samples
    print "S:", len(suscept_samples), "| R:", len(resist_samples)
    # Add the base model!!
    if "83332.12" not in list_of_random_items:
        list_of_random_items.append('83332.12')
    # Get labeled test data - specifically for isoniazid.
    drug_target = resistance_data.ix[list_of_random_items, drug_choice]
    # Remove strains that were not tested on the drug - have NaN
    drug_target.reset_index()
    drug_target_no_NaN = drug_target.reset_index().dropna().set_index('genome_id')
    # Convert R and S to integer values.
    drug_target_binary = drug_target_no_NaN[drug_choice].apply(lambda x: drug_type_binary(x))
    # Drop columns
    gene_vs_genome_frame = gene_vs_genome_frame.loc[:, (gene_vs_genome_frame.sum(axis=0) != len(list(gene_vs_genome_frame.index)))]
    gene_vs_genome_frame = gene_vs_genome_frame.loc[:, (gene_vs_genome_frame.sum(axis=0) != 0)]
    # Get the gene training dataframe + gene testing dataframe... as welll as for synthesis set
    gene_train = gene_vs_genome_frame.ix[list(drug_target_no_NaN.index), :]
    # RETURN
    return drug_target_binary, gene_train


def drug_resist_type(x):
    '''Takes rows in dataframe and creates new column '''
    '''MDR (multi drug resistant): resistance to at least isoniazid and rifampin'''
    '''XDR (extensively drug resistant): resistance to isoniazid and rifampin, plus any 
        fluoroquinolone and at least one of 3 injectaable second line drugs'''
    '''XXDR (totally drug resistant): resistant to all first and second line TB drugs'''
    # Multi Drug Resistant - MDR
    if x.isoniazid == 'R' and x.rifampicin == 'R':
        res_type = 'MDR'
    elif x.rifampicin == 'R':
        res_type = 'RR' # Rifampicin Resistant
    else:
        res_type = 'Susceptible'
    # All first line drugs
    if res_type == 'MDR' and x.ethambutol == "R" and x.pyrazinamide == "R": # and x.streptomycin == "R":
        all_first_line = "Yes"
    else:
        all_first_line = "No"
    # Second line aminoglycosides - at least 1.
    if x.amikacin == 'R' or x.kanamycin == 'R' or x.capreomycin == 'R':
        second_line = "Yes"
    else:
        second_line = "No"
    # Second line Flouroquinols - 
    if x.ciprofloxacin == 'R' or x.ofloxacin == 'R' or x.moxifloxacin == 'R':
        second_flouro = "Yes"
    else:
        second_flouro = "No"
    # ALL second line drugs
    if (x.ciprofloxacin == 'R' and x.ofloxacin == 'R' and x.moxifloxacin == 'R' and 
        x.amikacin == 'R' and x.kanamycin == 'R' and x.capreomycin == 'R' and x.cycloserine and
        x.ethionamide == 'R' and x.prothionamide == 'R' and x["4-aminosalicylic_acid"]== 'R'):
        all_second_line = "Yes"
    else:
        all_second_line = "No"
    # Extensively Drug Resistant - XDR
    if res_type == 'MDR' and second_line == "Yes" and second_flouro == "Yes":
        res_type = 'XDR'
    # Totally Drug Resistant - XXDR
    if all_first_line == "Yes" and all_second_line == "Yes":
        res_type = 'XXDR'
    #print x
    return res_type



def clster_identifier_func(clster_manual_map, sim_all_df_T, clust_to_rv, rv_to_name, clust_to_prod):
    
    clster_to_IDENTIFIER = {}
    for x in list(sim_all_df_T.index):
        clster_name = str(x).split("_")[0].replace("ter", "ter ")
        var_index = x.split("_")[1]
        if clster_name in clust_to_rv.keys():
            rv_id = clust_to_rv[clster_name]
            # rv_list.append(rv_id)
            if rv_id in rv_to_name.keys():
                rv_NAME = rv_to_name[rv_id]
                if rv_NAME!=None:
                    # rv_list.append(rv_id)
                    rv_NAME_final = rv_id + ", " + rv_NAME
                    # clster_to_name.update({x: rv_to_name[rv_id]})
                elif rv_id in clster_manual_map.keys():
                    rv_NAME_final = rv_id + ", " + clster_manual_map[rv_id]
                    # rv_list.append(rv_id)
                else:
                    rv_NAME_final = rv_id
                    # clster_to_name.update({x: rv_id})
            if clster_name in clust_to_prod.keys():
                rv_NAME_final = rv_NAME_final + ", " + clust_to_prod[clster_name]
        elif clster_name in clust_to_prod.keys():
            rv_NAME_final = clust_to_prod[clster_name]
        clster_to_IDENTIFIER.update({x: clster_name+rv_NAME_final+"_"+var_index})

    rv_list = []
    for x in list(sim_all_df_T.index):
        clster_name = str(x).split("_")[0].replace("ter", "ter ")
        if clster_name in clust_to_rv.keys():
            rv_id = clust_to_rv[clster_name]
            if rv_id != None:
                rv_list.append(rv_id)
            else:
                print "miss", clster_name
                
    return clster_to_IDENTIFIER, rv_list



def resist_percentage(resistance_data, list_of_strains, drug):
    return resistance_data.loc[list_of_strains,drug].sum()/float(len(resistance_data.loc[list_of_strains,drug].index))


def plot_svm_heatmap(tmp_db, fgsz, allele_to_colormap, cmap_choose):
    yticks = tmp_db.index
    tot_len_y = int(len(yticks))
    keptticks = yticks[::tot_len_y/tot_len_y]
    yticks = ['' for y in yticks]
    yticks[::tot_len_y/tot_len_y] = keptticks

    xticks = tmp_db.columns
    tot_len_x = int(len(xticks))
    keptticks = xticks[::tot_len_x/tot_len_x]
    xticks = ['' for y in xticks]
    xticks[::tot_len_x/tot_len_x] = keptticks
    
    g = sns.clustermap(tmp_db, method = "complete", metric="cityblock", 
                       yticklabels=True, xticklabels=False, 
                       cmap = cmap_choose,
                       linewidths=.1,
                       row_colors = allele_to_colormap,
                       col_colors = allele_to_colormap,
                       figsize=fgsz)
    # figsize=(22,4)
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0);
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90);
    return g


def svm_top_weights_colormap(sim_all_df_T, clster_to_IDENTIFIER, top_num, X1, y1, clster_var_mapping_prod):
    
    sim_all_df_Prod = sim_all_df_T.rename(index = clster_to_IDENTIFIER, inplace=False).sort_values("svm_weight_sum", ascending=False)[:top_num]
    sim_all_df_Prod.drop("svm_weight_sum", axis=1, inplace=True)

    # drug_name = "ofloxacin"

    allele_to_resist_dict = {}
    allele_to_colormap = []

    clust_alleles = list(sim_all_df_T.sort_values("svm_weight_sum", ascending=False)[:top_num].index)

    allele_df = X1.copy()[clust_alleles]
    suscept_y = y1[y1==0].copy()
    resist_y = y1[y1==1].copy()
    suscept_alleles = allele_df.loc[suscept_y.index, :].copy()
    resist_alleles = allele_df.loc[resist_y.index, :].copy()


    for allele in resist_alleles.columns:
        try:
            allele_r_percent = round(resist_alleles[allele].sum()/float(allele_df[allele].sum()),2)
            allele_to_resist_dict.update({clster_var_mapping_prod[allele]: allele_r_percent})
            # allele_to_colormap.append(plt.cm.Spectral(allele_r_percent))
            # allele_to_colormap.append(plt.cm.gist_yarg(allele_r_percent))
            # allele_to_colormap.append(plt.cm.binary(allele_r_percent))
            # allele_to_colormap.append(plt.cm.coolwarm(allele_r_percent))

            allele_to_colormap.append(sns.diverging_palette(250, 10, sep=60, n=100)[int(allele_r_percent*100-1)])

        except:
            print allele,drug_name,"ZeroDivisionError: float division by zero"
    
    return sim_all_df_Prod, allele_to_colormap, clust_alleles



def get_MDR_training(resistance_data, data_to_plot, MDR_or_XDR):
    # MDR
    resist_type = resistance_data.apply(lambda x: drug_resist_type(x) , axis=1)

    print "# RR", len(resist_type[resist_type=="RR"].index)
    print "# MDR", len(resist_type[resist_type=="MDR"].index)
    print "# XDR", len(resist_type[resist_type=="XDR"].index)

    resist_type[resist_type=="Susceptible"] = 0
    resist_type[resist_type=="RR"] = 0
    if MDR_or_XDR == "XDR":
        resist_type[resist_type=="MDR"] = 0
        resist_type[resist_type=="XDR"] = 1
    if MDR_or_XDR == "MDR":
        resist_type[resist_type=="MDR"] = 1
        resist_type[resist_type=="XDR"] = 1
    # resist_type[resist_type<1] = 0
    print "number of ",MDR_or_XDR," ",resist_type.sum()

    shared_strains = set(list(data_to_plot.index)).intersection(set(resist_type.index))
    print "number of strains available for classification",len(shared_strains)

    y_MDR = resist_type[list(shared_strains)]
    print y_MDR.sum()

    X = data_to_plot.loc[shared_strains, :].copy()
    y = y_MDR.fillna(0).copy()
    drug_name = "MDR"
    return X, y
    
# X1, y1 = get_MDR_training(resistance_data, data_to_plot, MDR_or_XDR="XDR")


def save_ROC_curve_figure(drug_name, fpr, tpr, roc_auc, simulation_iterations, estimator_type, save_loc):
    plt.style.use('seaborn-white')
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for t in range(simulation_iterations):
        plt.plot(fpr[t], tpr[t])

    avg_auc_string = str(round(np.mean(roc_auc.values()), 2)*100)[:2]

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title(drug_name+"_"+estimator_type+"|| ROC Curves: Avg AUC = "+avg_auc_string+"%")
    plt.savefig(save_loc+"ROC_"+drug_name+"_"+estimator_type+"_"+avg_auc_string+".png")
    plt.show()

    
# -------------------------------------------------------------------------------------------------------
# ------------------------------------------ Allele Co-occurence functions ------------------------------
# -------------------------------------------------------------------------------------------------------
def explain_cluster_AMR(clust_explain, All_or_OutDf, y_drug, X_alleles, cl_vr_df, fin_vr_list, verbose_out):
    
    if All_or_OutDf == "All":
        clst_alleles = [x for x in list(set(cl_vr_df.columns)) if clust_explain in x]
    else:
        clst_alleles = [x for x in list(set(fin_vr_list)) if clust_explain in x]
    
    if verbose_out==True:
        print clst_alleles
    allele_df = X_alleles.copy()[clst_alleles]
    
    suscept_y = y_drug[y_drug==0].copy()
    resist_y = y_drug[y_drug==1].copy()
    suscept_alleles = allele_df.loc[suscept_y.index, :].copy()
    resist_alleles = allele_df.loc[resist_y.index, :].copy()
    
    allele_to_resistString_dict = {}
    allele_to_resistPercent_dict = {}
    
    for allele in resist_alleles.columns:
        try:
            if verbose_out==True:
                print allele,"|| # of strains =",allele_df[allele].sum(),"|| # of R =",resist_alleles[allele].sum(),
                print "|| R% =",round(resist_alleles[allele].sum()/float(allele_df[allele].sum()),2), ",",
                print str(resist_alleles[allele].sum())+"/"+str(round(float(allele_df[allele].sum()),2)),
                print "|| S% =",round(suscept_alleles[allele].sum()/float(allele_df[allele].sum()), 2), ",",
                print str(suscept_alleles[allele].sum())+"/"+str(round(float(allele_df[allele].sum()),2))

            resist_string = str(int(resist_alleles[allele].sum()))+"/"+str(int(allele_df[allele].sum()))
            resist_string = "("+resist_string +")"+ ": " + str(round(resist_alleles[allele].sum()/float(allele_df[allele].sum()),2))

            allele_to_resistString_dict.update({allele: resist_string})
            allele_to_resistPercent_dict.update({allele: round(resist_alleles[allele].sum()/float(allele_df[allele].sum()),2)})
        except:
            print allele,"ZeroDivisionError: float division by zero"
        
    return allele_to_resistString_dict, allele_to_resistPercent_dict



def explain_TWOcluster_AMR(clust_explain_1, clust_explain_2, All_or_OutDf, y_drug, X_alleles, cl_vr_df, fin_vr_list, verbose_out):
    
    if All_or_OutDf == "All":
        clst_alleles_1 = [x for x in list(set(cl_vr_df.columns)) if clust_explain_1 in x]
        clst_alleles_2 = [x for x in list(set(cl_vr_df.columns)) if clust_explain_2 in x]
    else:
        clst_alleles_1 = [x for x in list(set(fin_vr_list)) if clust_explain_1 in x]
        clst_alleles_2 = [x for x in list(set(fin_vr_list)) if clust_explain_2 in x]
        
    # print clst_alleles_1, clst_alleles_2
    both_clust_alleles = clst_alleles_1 + clst_alleles_2
    allele_df = X_alleles.copy()[both_clust_alleles]
    
    suscept_y = y_drug[y_drug==0].copy()
    resist_y = y_drug[y_drug==1].copy()
    suscept_alleles = allele_df.loc[suscept_y.index, :].copy()
    resist_alleles = allele_df.loc[resist_y.index, :].copy()
    
    double_allele_resist_dict = {}
    double_allele_resistPercent_dict = {}
    if verbose_out == True:
        print clst_alleles_1
    
    for allele_1 in clst_alleles_1:
        double_allele_resist_dict.update({allele_1: {}})
        double_allele_resistPercent_dict.update({allele_1: {}})
        for allele_2 in clst_alleles_2:

            resist_allele_var_1 = resist_alleles[resist_alleles.loc[: , allele_1]==1]
            resist_allele_var_both = resist_allele_var_1[resist_allele_var_1.loc[:, allele_2]==1]

            all_allele_var_1 = allele_df[allele_df.loc[:, allele_1]==1]
            all_allele_var_both = all_allele_var_1[all_allele_var_1.loc[:, allele_2]==1]
            
            resist_strain_num = len(resist_allele_var_both)
            all_strain_num = len(all_allele_var_both.index)
            
            try:
                if resist_strain_num < 1:
                    if all_strain_num>0:
                        if verbose_out==True:
                            print allele_1, allele_2, 
                            print "| # of strains = ", all_strain_num, 
                            print "| # of R = ", resist_strain_num,
                            print "| %R: ", "0.0"
                        entry_string = "("+str(all_strain_num)+"/"+str(resist_strain_num)+")"+": 0.0"
                        double_allele_resist_dict[allele_1].update({allele_2: entry_string})
                    double_allele_resist_dict[allele_1].update({allele_2: "-"})
                    double_allele_resistPercent_dict[allele_1].update({allele_2: 0})
                else:
                    resist_percent = round(resist_strain_num/float(all_strain_num),2)
                    if verbose_out==True:
                        print allele_1, allele_2, 
                        print "| # of strains = ", all_strain_num, 
                        print "| # of R = ", resist_strain_num,
                        print "| %R: ", resist_percent
                    entry_string = "("+str(resist_strain_num)+"/"+str(all_strain_num)+")"+": "+str(resist_percent)
                    double_allele_resist_dict[allele_1].update({allele_2: entry_string})
                    double_allele_resistPercent_dict[allele_1].update({allele_2: resist_percent})
            except:
                print allele_1, allele_2,"ZeroDivisionError: float division by zero"
    return double_allele_resist_dict, double_allele_resistPercent_dict


def get_2allele_resist_dataframes(clust1, clust2, drug_name, All_or_OutDf, y1, cv_df_y1, cv_df, clust_alleles, verbose_out, out_loc):
    
    double_allele_resist_dict, double_allele_resistPercent_dict = explain_TWOcluster_AMR(clust1, clust2, 
                                   All_or_OutDf, y1, cv_df_y1, cv_df, clust_alleles, verbose_out)
    
    clust_1_resist_dict, clust_1_resistPercent_dict = explain_cluster_AMR(clust1,All_or_OutDf, y1, cv_df_y1, 
                        cv_df, clust_alleles, verbose_out)
    clust_1_df = pd.DataFrame.from_dict(clust_1_resist_dict, orient="index")
    clust_1_df.columns = [clust1]
    clust_1percent_df = pd.DataFrame.from_dict(clust_1_resistPercent_dict, orient="index")
    clust_1percent_df.columns = [clust1]

    clust_2_resist_dict, clust_2_resistPercent_dict = explain_cluster_AMR(clust2,All_or_OutDf, y1, cv_df_y1, 
                        cv_df, clust_alleles, verbose_out)
    clust_2_df = pd.DataFrame.from_dict(clust_2_resist_dict, orient="index")
    clust_2_df.columns = [clust2]
    clust_2percent_df = pd.DataFrame.from_dict(clust_2_resistPercent_dict, orient="index")
    clust_2percent_df.columns = [clust2]

    two_cluster_df = pd.DataFrame.from_dict(double_allele_resist_dict)
    two_cluster_df = pd.concat([two_cluster_df, clust_2_df], axis=1)
    two_cluster_df = two_cluster_df.append(clust_1_df.T)

    two_clusterPercent_df = pd.DataFrame.from_dict(double_allele_resistPercent_dict)
    two_clusterPercent_df = pd.concat([two_clusterPercent_df, clust_2percent_df], axis=1)
    two_clusterPercent_df = two_clusterPercent_df.append(clust_1percent_df.T)
    
    output = drug_name+"__"+clust1+clust2
    print output

    writer = pd.ExcelWriter(out_loc+output+".xlsx")
    two_cluster_df.to_excel(writer, sheet_name='resist_fraction', index=True)
    two_clusterPercent_df.to_excel(writer, sheet_name='resist_percent', index=True)
    writer.save()
    return two_cluster_df, two_clusterPercent_df



def allele_cooccurence_pValue(all_1, all_2, y_drg, X_all, association_test):
    """
    Takes in two alleles, creates a cooccurence column, and computes the association of that
    column with the AMR phenotype.
    """
    two_allele_df = X_all.loc[:, [all_1,all_2]].copy()
    two_allele_df["cooccurence"] = two_allele_df[all_1] + two_allele_df[all_2]
    two_allele_df["cooccurence"][two_allele_df["cooccurence"]<2]=0
    two_allele_df["cooccurence"][two_allele_df["cooccurence"]==2]=1
    
    if association_test == "chi2":
        test_stats, pVals = chi2(two_allele_df, y_drg)
    elif association_test == "f_classif":
        test_stats, pVals = f_classif(two_allele_df, y_drg)
        
    allele_cooccurence_pVal_dict = {}
    for col_ind in range(len(two_allele_df.columns)):
        allele_cooccurence_pVal_dict.update({two_allele_df.columns[col_ind]: 
                                             {"test_stat": test_stats[col_ind], "pVal": pVals[col_ind]}})
    
    return allele_cooccurence_pVal_dict


def return_2allele_Pval_df(assoc_test_type, two_clusterPercent_df, two_clst_df, X_allez, y1):
    two_clusterPval_df = two_clusterPercent_df.copy()
    for cl1, row in two_clusterPercent_df.iterrows():
        for cl2 in row.index: 
    #         print cl1, cl2
    #         print len(cl1.split("_")), len(cl2.split("_")[1])

            if cl1.split("_")[0] != cl2.split("_")[0]:
                if len(cl1.split("_")[1])==0:
                    if len(cl2.split("_")[1])>0:
                        allele_pVal_dict = allele_cooccurence_pValue(cl1, cl2, y1, 
                                                                     X_allez, assoc_test_type)
                        two_clusterPval_df.loc[cl1, cl2] = allele_pVal_dict[cl2]["pVal"]

                elif len(cl2.split("_")[1])==0:
                    if len(cl1.split("_")[1])>0:
                        allele_pVal_dict = allele_cooccurence_pValue(cl1, cl2, y1, 
                                                                     X_allez,assoc_test_type)
                        two_clusterPval_df.loc[cl1, cl2] = allele_pVal_dict[cl1]["pVal"]

                else:
                    if two_clst_df.loc[cl1, cl2] != "-":
                        allele_pVal_dict = allele_cooccurence_pValue(cl1, cl2, y1, 
                                                                     X_allez,assoc_test_type) 
                        two_clusterPval_df.loc[cl1, cl2] = allele_pVal_dict["cooccurence"]["pVal"]

            elif cl1.split("_")[0] == cl2.split("_")[0] and len(cl2.split("_")[1])>0:
                if assoc_test_type == "chi2":
                    test_single, pVal_single = chi2(X_allez[[cl2, cl2]], y1)
                elif assoc_test_type == "f_classif":
                    test_single, pVal_single = f_classif(X_allez[[cl2, cl2]], y1)
                two_clusterPval_df.loc[cl1, cl2] = pVal_single[0]

            elif cl1.split("_")[0] == cl2.split("_")[0] and len(cl1.split("_")[1])>0:
                if assoc_test_type == "chi2":
                    test_single, pVal_single = chi2(X_allez[[cl1, cl1]], y1)
                elif assoc_test_type == "f_classif":
                    test_single, pVal_single = f_classif(X_allez[[cl1, cl1]], y1)
                two_clusterPval_df.loc[cl1, cl2] = pVal_single[0]
                
    return two_clusterPval_df


def allele_cooccurence_oddsRatio(all_1, all_2, y_drg, X_all, add_small_value_to_table, log_or_not, single_or_double_tuple):
    """
    Takes in two alleles, creates a cooccurence column, and computes the association of that
    column with the AMR phenotype.
    """
    # print X_all.shape, all_1, all_2, X_all.columns
    two_allele_df = X_all.loc[:, [all_1,all_2]].copy()
    
    if single_or_double_tuple[0] == "double":
        two_allele_df["cooccurence"] = two_allele_df[all_1] + two_allele_df[all_2]
        two_allele_df["cooccurence"][two_allele_df["cooccurence"]<2]=0
        two_allele_df["cooccurence"][two_allele_df["cooccurence"]==2]=1
    elif single_or_double_tuple[0] == "single" and all_1 == all_2:
        two_allele_df = X_all.loc[:, [all_2]].copy()
        two_allele_df["cooccurence"] = two_allele_df[single_or_double_tuple[1]]
    elif single_or_double_tuple[0] == "single":
        two_allele_df["cooccurence"] = two_allele_df[single_or_double_tuple[1]]
    
    two_allele_df["resistant"] = y_drg
    
    cooccurence_resistant = float(len(two_allele_df[(two_allele_df["cooccurence"]==1) & (two_allele_df["resistant"]==1)].index))
    cooccurence_susceptible = float(len(two_allele_df[(two_allele_df["cooccurence"]==1) & (two_allele_df["resistant"]==0)].index))
    
    no_cooccurence_resistant = float(len(two_allele_df[(two_allele_df["cooccurence"]==0) & (two_allele_df["resistant"]==1)].index))
    no_cooccurence_susceptible = float(len(two_allele_df[(two_allele_df["cooccurence"]==0) & (two_allele_df["resistant"]==0)].index))
    
    # add_small_value_to_table = .1
    # if one of the cells is 0, there won't be an odds ratio. To get past this, we add a small value of .5 to each value
    if cooccurence_resistant==0 or cooccurence_susceptible==0 or no_cooccurence_resistant==0 or no_cooccurence_susceptible==0:
        cooccurence_resistant+=add_small_value_to_table
        cooccurence_susceptible+=add_small_value_to_table
        no_cooccurence_resistant+=add_small_value_to_table
        no_cooccurence_susceptible+=add_small_value_to_table
    
    odds_ratio = (cooccurence_resistant/cooccurence_susceptible)/(no_cooccurence_resistant/no_cooccurence_susceptible)
    
    if log_or_not == True:
        odds_ratio = np.log(odds_ratio)
    
    return odds_ratio


def allele_cooccurence_oddsRatio_xNOTy(all_1, all_2, y_drg, X_all, add_small_value_to_table, log_or_not):
    """
    Takes in two alleles, creates a cooccurence column, and computes the association of that
    column with the AMR phenotype.
    """
    # print X_all.shape, all_1, all_2, X_all.columns
    two_allele_df = X_all.loc[:, [all_1, all_2]].copy()
    two_allele_df["resistant"] = y_drg
    
    cooccurence_resistant_x1_NOT_y1 = float(len(two_allele_df[(two_allele_df[all_1]==1) & (two_allele_df[all_2]==0) & (two_allele_df["resistant"]==1)].index))
    cooccurence_susceptible_x1_NOT_y1 = float(len(two_allele_df[(two_allele_df[all_1]==1) & (two_allele_df[all_2]==0) & (two_allele_df["resistant"]==0)].index))

    cooccurence_resistant_NOT_x1_NOT_y1 = float(len(two_allele_df[(two_allele_df[all_1]==0) & (two_allele_df[all_2]==0) & (two_allele_df["resistant"]==1)].index))
    cooccurence_susceptible_NOT_x1_NOT_y1 = float(len(two_allele_df[(two_allele_df[all_1]==0) & (two_allele_df[all_2]==0) & (two_allele_df["resistant"]==0)].index))

    if cooccurence_resistant_x1_NOT_y1==0 or cooccurence_susceptible_x1_NOT_y1==0 or cooccurence_resistant_NOT_x1_NOT_y1==0 or cooccurence_susceptible_NOT_x1_NOT_y1==0:
        cooccurence_resistant_x1_NOT_y1+=add_small_value_to_table
        cooccurence_susceptible_x1_NOT_y1+=add_small_value_to_table
        cooccurence_resistant_NOT_x1_NOT_y1+=add_small_value_to_table
        cooccurence_susceptible_NOT_x1_NOT_y1+=add_small_value_to_table

    odds_ratio = (cooccurence_resistant_x1_NOT_y1/cooccurence_susceptible_x1_NOT_y1)/(cooccurence_resistant_NOT_x1_NOT_y1/cooccurence_susceptible_NOT_x1_NOT_y1)
    if log_or_not == True:
        odds_ratio = np.log(odds_ratio)
    
    return odds_ratio


def return_2allele_oddsRatio_df(two_clusterPercent_df, two_clst_df, X_allez, y1, add_smallval_table, log_or_no):
    two_clusterOddsRatio_df = two_clusterPercent_df.copy()
    for cl1, row in two_clusterPercent_df.iterrows():
        for cl2 in row.index: 
            if cl1.split("_")[0] != cl2.split("_")[0]:
            
            # ----- case: (clusterX_1, clusterY_1) -----
                if two_clst_df.loc[cl1, cl2] != "-" and (len(cl1.split("_")[1])>0 or len(cl2.split("_")[1])>0):
                    # print cl1, cl2, "_3"
                    allele_oddsRatio = allele_cooccurence_oddsRatio(cl1, cl2, y1, X_allez, 
                                                                    add_smallval_table, log_or_no, ("double", "-"))
                    two_clusterOddsRatio_df.loc[cl1, cl2] = allele_oddsRatio

            elif cl1.split("_")[0] == cl2.split("_")[0] and len(cl2.split("_")[1])>0:
                
                # ----- case: (clusterX_, clusterX_1) -----
                # print cl1, cl2, "_4"
                allele_oddsRatio = allele_cooccurence_oddsRatio(cl1, cl2, y1, X_allez, 
                                                                add_smallval_table, log_or_no, ("single", cl2))
                two_clusterOddsRatio_df.loc[cl1, cl2] = allele_oddsRatio

            elif cl1.split("_")[0] == cl2.split("_")[0] and len(cl1.split("_")[1])>0:
                
                # ----- case: (clusterY_1, clusterY_) -----
                # print cl1, cl2, "_5"
                allele_oddsRatio = allele_cooccurence_oddsRatio(cl1, cl2, y1, X_allez, 
                                                                add_smallval_table, log_or_no, ("single", cl1))
                two_clusterOddsRatio_df.loc[cl1, cl2] = allele_oddsRatio
                
    return two_clusterOddsRatio_df



def make_table_figure(clust1, clust2, drg_name, two_cluster_df, two_clusterPercent_df, two_varPval_df, two_varOdds_df, clust_to_rv, rv_to_name, clster_manual_map, which_df_2_color, out_loc, color_bar_or_not):
    """
    Function for generating allele co-occurence table figures.
    """
    rv_name1, rv_name2 = clust_to_rv[clust1.strip("_").replace("ter", "ter ")], clust_to_rv[clust2.strip("_").replace("ter", "ter ")]
    name1, name2 = rv_to_name[rv_name1], rv_to_name[rv_name2]

    if name1 == None and rv_name1 in clster_manual_map.keys():
        name1 = clster_manual_map[rv_name1]
    elif name1 == None:
        name1 = rv_name1

    if name2 == None and rv_name2 in clster_manual_map.keys():
        name2 = clster_manual_map[rv_name2]
    elif name2 == None:
        name2 = rv_name2
        
    # ------ Make styled dataframe by changing the index and column names, moving shit around, etc.... ------
    if which_df_2_color == "percent":
        save_occurance_df = two_clusterPercent_df.copy()
        vmin_bot=0.2
        vmax_top=0.95
    elif which_df_2_color == "pVal":
        save_occurance_df = two_varPval_df.copy()
        save_occurance_df = -np.log(save_occurance_df).replace([np.inf, -np.inf], np.nan)
        vmin_bot=0
        vmax_top=30
    elif which_df_2_color == "oddsRatio":
        save_occurance_df = two_varOdds_df.copy()
        abs_max_bound = max(abs(save_occurance_df.fillna(0).values.min()), save_occurance_df.fillna(0).values.max())
        vmin_bot = -abs_max_bound
        vmax_top = abs_max_bound

    new_cols = [str(x.split("_")[1]) if len(str(x.split("_")[1]))>0 else x for x in save_occurance_df.columns]
    new_inds = [str(x.split("_")[1]) if len(str(x.split("_")[1]))>0 else x for x in save_occurance_df.index]

    new_cols_Name = [str(x.split("_")[1]) if len(str(x.split("_")[1]))>0 else rv_name2 for x in save_occurance_df.columns]
    new_inds_Name = [str(x.split("_")[1]) if len(str(x.split("_")[1]))>0 else rv_name1 for x in save_occurance_df.index]

    allele_cols = [x for x in new_cols if "Cluster" not in x]
    # cluster_col = [x for x in new_cols if "Cluster" in x]
    cluster_col = [rv_name2]

    allele_ind = [x for x in new_inds if "Cluster" not in x]
    # cluster_ind = [x for x in new_inds if "Cluster" in x]
    cluster_ind = [rv_name1]

    # save_occurance_df.columns = new_cols
    # save_occurance_df.index = new_inds
    save_occurance_df.columns = new_cols_Name
    save_occurance_df.index = new_inds_Name

    save_occurance_df = save_occurance_df[allele_cols+cluster_col]
    save_occurance_df = save_occurance_df.reindex(index = allele_ind+cluster_ind)

    labeled_two_cluster_df = two_cluster_df.copy()
    # labeled_two_cluster_df.columns = new_cols
    # labeled_two_cluster_df.index = new_inds
    labeled_two_cluster_df.columns = new_cols_Name
    labeled_two_cluster_df.index = new_inds_Name

    labeled_two_cluster_df = labeled_two_cluster_df[allele_cols+cluster_col]
    labeled_two_cluster_df = labeled_two_cluster_df.reindex(index = allele_ind+cluster_ind)

    # ---- Add the two columns and rows to both ends of "save_occurance_df"and "labeled_two_cluster_df"
    # --- Add the two columns to right side of dataframe ----
    
    resist_vector_name = "#R"
    total_vector_name = "Total"
    
    labeled_two_cluster_df[resist_vector_name] = labeled_two_cluster_df.fillna("-")[cluster_col[0]].apply(lambda x: x.split(":")[0].strip("()").split("/")[0] if "/" in x else x)
    labeled_two_cluster_df[total_vector_name] = labeled_two_cluster_df.fillna("-")[cluster_col[0]].apply(lambda x: x.split(":")[0].strip("()").split("/")[1] if "/" in x else x)
    
    # --- Add the two rows to the bottom of dataframe ----
    row_resistant = labeled_two_cluster_df.fillna("-").loc[cluster_ind[0], :].apply(lambda x: x.split(":")[0].strip("()").split("/")[0] if "/" in x else x)
    row_resistant.name = resist_vector_name
    row_total = labeled_two_cluster_df.fillna("-").loc[cluster_ind[0], :].apply(lambda x: x.split(":")[0].strip("()").split("/")[1] if "/" in x else x)
    row_total.name = total_vector_name
    labeled_two_cluster_df = labeled_two_cluster_df.append(row_resistant)
    labeled_two_cluster_df = labeled_two_cluster_df.append(row_total)

    # --- Drop Gene ID row ---
    labeled_two_cluster_df.drop(cluster_col, axis=1, inplace=True)
    labeled_two_cluster_df.drop(cluster_ind, axis=0, inplace=True)
    
    # ---- Change save_occurance_df -----
    save_occurance_df[resist_vector_name] = save_occurance_df[cluster_col[0]]
    save_occurance_df[total_vector_name] = save_occurance_df[cluster_col[0]]
    row_occurTotal_add = save_occurance_df.loc[cluster_ind[0], :].copy()
    row_occurTotal_add.name = total_vector_name
    row_occurR_add = save_occurance_df.loc[cluster_ind[0], :].copy()
    row_occurR_add.name = resist_vector_name
    save_occurance_df = save_occurance_df.append(row_occurR_add)
    save_occurance_df = save_occurance_df.append(row_occurTotal_add)
    save_occurance_df.drop([cluster_col[0]], axis=1, inplace=True)
    save_occurance_df.drop([cluster_ind[0]], inplace=True)

    # ------ Make seaborn heatmap from the two dataframes that looks like a table ----
    cm = sns.diverging_palette(236, 15, s=86, l=65, sep=10, n=9, as_cmap=True)
    
    # ------ reorder the columns and index based on the numeric value instead of string ------
    ordered_cols = [int(x) if x.isdigit() else x for x in save_occurance_df.columns]
    ordered_cols.sort()
    ordered_cols_string = [str(x) for x in ordered_cols]
    ordered_inds = [int(x) if x.isdigit() else x for x in save_occurance_df.index]
    ordered_inds.sort()
    ordered_inds_string = [str(x) for x in ordered_inds]

    save_occurance_df = save_occurance_df[ordered_cols_string]
    save_occurance_df.reindex(ordered_inds_string)
    
    labeled_two_cluster_df = labeled_two_cluster_df[ordered_cols_string]
    labeled_two_cluster_df.reindex(ordered_inds_string)
    
    labels = np.array(labeled_two_cluster_df.fillna("-").astype(str).copy())
    only_frac_labels = []
    for x in labels:
        row_labels = [y.split(":")[0].strip("()") for y in x]
        row_labels = ['$\\dfrac{{{0}}}{{{1}}}$'.format(y.split("/")[0], y.split("/")[1]) if "/" in y else y for y in row_labels ]
        only_frac_labels.append(row_labels)

        
    if color_bar_or_not == True:
        plt.figure(figsize=(len(save_occurance_df.columns)*.65, len(save_occurance_df.index)*.5))
    else:
        plt.figure(figsize=(len(save_occurance_df.columns)*.5, len(save_occurance_df.index)*.5))
    g = sns.heatmap(save_occurance_df, annot=np.array(only_frac_labels),fmt = '',
                    cmap=cm, vmin=vmin_bot, vmax=vmax_top,
                    cbar=True, linewidths=0.6, linecolor="black", 
                    annot_kws={"color":"black"}, 
                    # annot_kws={"size": rowCol_ratio_in_figure*3, "color":"black"},
                    square=True);
                    # 
    g.set_xlabel(name1+" alleles", rotation=0); # , size=16
    g.set_ylabel(name2+"\nalleles", rotation=0); # , size=16
    g.xaxis.tick_top()
    g.xaxis.set_label_position('top') 
    g.tick_params(axis="both", which="major", length=0)
    plt.setp(g.yaxis.get_majorticklabels(), rotation=0); # , size=10
    plt.setp(g.xaxis.get_majorticklabels(), rotation=0); # , size=10
    plt.savefig(out_loc+drg_name+"_"+name1+"__"+name2+"_"+which_df_2_color+"_occurance.svg",
               bbox_inches="tight", dpi=200)
    plt.close()
    return save_occurance_df, labeled_two_cluster_df
    
# -------------------------------------------------------------------------------------------------------
# ------------------------------------------ General Statistical Test functions -------------------------
# -------------------------------------------------------------------------------------------------------
def get_enrichment_pvals(all_annotation_counter, subset_annotation_counter, pval_thresh):
    """
    all_annotation_counter
            - Counter object of the list of all annotations for all feature
            - i.e. for all the possible genes, the Counter of all possible associated pathways (non-unique)
    subset_annotation_counter
            - Counter object of the subset to determine enrichment for.
    pval_thresh
            - The p-val cutoff for what determines significance or not
    """
    
    annot_pVal_dict = {}
    
    for annot_of_interest in subset_annotation_counter.keys():
        # subsyst_of_interest = "Citric Acid Cycle"

        # ---- M = Total number of Reactions ----
        M = sum(all_annotation_counter.values())

        # ---- n = Number of Reactions with subsystem of interest ----
        n = all_annotation_counter[annot_of_interest]

        # ---- N = Subset number of Reactions in top PCA loadings with subsystem of interest ----
        N = sum(subset_annotation_counter.values())

        # ---- k = Subset number of Reactions in top PCA loadings ----
        k = subset_annotation_counter[annot_of_interest]

        # print M, n, N, k
        pval = ss.hypergeom.sf(k, M, n, N)
        # p = hpd.pmf(k)
        if pval < pval_thresh:
            annot_pVal_dict.update({annot_of_interest: pval})
        
    return annot_pVal_dict