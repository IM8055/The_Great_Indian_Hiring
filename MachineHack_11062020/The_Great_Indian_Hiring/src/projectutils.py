import pandas as pd


class clsProjectUtilityFunction:

    @staticmethod
    def funcColumnAggregation(fncpdf, fncpcolumn, fncpNumericCol=[], fncpCategoricalCol=[], fncpmonthcol=[]):
        """
        This function creates and returns a aggregated dataframe based on a specific column

        :fncpcolumn - Column for which we need to apply groupby function
        :fncpdf - DataFrame where the column is present
        :fncpNumericCol - Numeric column for which aggregation should be done.
        :fncpmonthcol -  Month column for which aggregation should be done.
        :fncpweekofyearcol - Weekofyear column for which aggregation should be done.

        :output - dataframe
        """
        dictAggs = {fncpcolumn: ['size', 'nunique']}
        if len(fncpNumericCol) >= 1:
            for col_ in fncpNumericCol:
                dictAggs[col_] = ['sum', 'min', 'max', 'mean']
        if len(fncpCategoricalCol) >= 1:
            for col_ in fncpCategoricalCol:
                dictAggs[col_] = ['size', 'nunique']
        if len(fncpmonthcol) >= 1:
            for col_ in fncpmonthcol:
                dictAggs[col_] = ['nunique', 'mean']
        aggsDF = fncpdf.groupby(fncpcolumn).agg(dictAggs)
        aggsDF = aggsDF.reset_index()
        aggsDF.columns = ['f_Grp'+fncpcolumn+'_'+'_'.join(x) if x not in fncpdf.columns else x for x in aggsDF.columns.ravel()]
        aggsDF = aggsDF.rename(columns={'f_Grp'+fncpcolumn+'_'+fncpcolumn+'_': fncpcolumn})
        aggsDF = fncpdf.merge(aggsDF, how='left', on=fncpcolumn)

        return aggsDF

    @staticmethod
    def funcMergingDFBasedOnYear(fncpdf, fncplistyear=[], fncplistmergecolumnon=[], fncpjointype='inner', fncplistcol=[]):
        """
        This function splits a dataframe based on year column and returns a merged dataframe with only unique
        combinations.

          :fncp - Dataframe that needs to be split based on year
          :fncp - List of years based on which we need to split the dataframe
          :fncpmergecol - List of columns based on which merging should happen
          :fncpjointype - Type of join to merge the dataframes
          :fncplistcol - List of columns that needs to be present in the new dataframe for every year
        """
        mergedDF = pd.DataFrame()
        fncplistTempCol = ['StockCode', 'f_InvoiceDateYear', 'UnitPrice']
        dictTempRename = {}
        for year_ in fncplistyear:
            for col_ in fncplistcol:
                if col_ not in fncplistmergecolumnon:
                    dictTempRename[col_] = col_ + '_' + str(year_)
            filteredDF = fncpdf.loc[fncpdf.f_InvoiceDateYear == year_, fncplistcol].rename(columns=dictTempRename)
            dictTempRename = {}
            if mergedDF.shape[0] == 0:
                mergedDF = filteredDF.copy()
            else:
                mergedDF = mergedDF.merge(filteredDF, how=fncpjointype, on=fncplistmergecolumnon)

        return mergedDF.drop_duplicates()
