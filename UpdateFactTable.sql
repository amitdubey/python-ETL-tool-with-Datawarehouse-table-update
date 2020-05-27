
##################Note to Self ###########
# Author  : Amit Dubey
# Date : May10th,2020
#Purpose : Dimension table insert script and update to agrregated Fact table in the data warehouse from OLTP --> OLAP DB
###############################################


USE testdb;

DROP TEMPORARY TABLE if exists FactTbl_may5_temp;

#create a temporary fact table schema for staging
CREATE TEMPORARY TABLE 
IF NOT EXISTS FactTbl_may5_temp
AS (SELECT * FROM FactTbl_may5 WHERE 1 = 0);

#Step 2. Populate The Temp  fact table from for a given time period 
INSERT INTO FactTbl_may5_temp
SELECT 
			ifnull(f.dateid ,0),
			f.CalendarDate ,
			f.CalendarQuarter ,
			f.YearMonth ,
			f.SubsidiaryCode ,
			f.SubsidiaryName ,
			f.TPId ,
			f.TPNamee ,
			f.DeviceName,
			f.PFAMName ,
			f.ProductPartNbr ,
			f.SellThruQTY ,
			f.SellinQTY ,
			f.Year ,
			f.Month ,
			f.Day ,
			f.Day_of_Week ,
			f.COMM_IND ,
			f.Bundle_IND,
			f.Processor ,
			f.RAM ,
			f.QTR 
			FROM FactTbl_may5 f
LEFT JOIN DimTbl_may5 d ON DATE_FORMAT(DATE(f.dateid) ,'%Y%m%d')  = DATE_FORMAT(date(CURDATE()) ,'%Y%m%d') ;
	

SELECT  *
FROM
    FactTbl_may5_temp
WHERE
    dateid = 0;

UPDATE FactTbl_may5 f
	INNER JOIN
		FactTbl_may5_temp t ON f.dateid = t.dateid
	LEFT JOIN
		DimTbl_may5 d ON f.dateid 
	SET 
			f.dateid = t.dateid,
			f.CalendarDate = t.CalendarDate,
			f.CalendarQuarter = t.CalendarQuarter,
			f.YearMonth = t.YearMonth,
			f.SubsidiaryCode = t.SubsidiaryCode,
			f.SubsidiaryName = t.SubsidiaryName,
			f.TPId = t.TPId,
			f.TPNamee = t.TPNamee,
			f.DeviceName = t.DeviceName,
			f.PFAMName = t.PFAMName,
			f.ProductPartNbr = t.ProductPartNbr,
			f.SellThruQTY = t.SellThruQTY,
			f.SellinQTY = t.SellinQTY,
			f.Year = t.Year,
			f.Month = t.Month,
			f.Day = t.Day,
			f.Day_of_Week = t.Day_of_Week,
			f.COMM_IND = t.COMM_IND,
			f.Bundle_IND = t.Bundle_IND,
			f.Processor = t.Processor,
			f.RAM = t.RAM,
			f.QTR = t.QTR
	WHERE
			t.dateid <> 0
        AND (
					f.CalendarDate <> t.CalendarDate
					OR f.CalendarQuarter <> t.CalendarQuarter
					OR f.YearMonth <> t.YearMonth
					OR f.SubsidiaryCode <> t.SubsidiaryCode
					OR f.SubsidiaryName <> t.SubsidiaryName
					OR f.TPId <> t.TPId
					OR f.TPNamee <> t.TPNamee
					OR f.DeviceName <> t.DeviceName
					OR f.PFAMName <> t.PFAMName
					OR f.ProductPartNbr <> t.ProductPartNbr
					OR f.SellThruQTY <> t.SellThruQTY
					OR f.SellinQTY <> t.SellinQTY
					OR f.Year <> t.Year
					OR f.Month <> t.Month
					OR f.Day <> t.Day
					OR f.Day_of_Week <> t.Day_of_Week
					OR f.COMM_IND <> t.COMM_IND
					OR f.Bundle_IND <> t.Bundle_IND
					OR f.Processor <> t.Processor
					OR f.RAM <> t.RAM
					OR f.QTR <> t.QTR);

#Step 4. Insert New Records in fact table
INSERT INTO FactTbl_may5
SELECT
			f.dateid,
			f.CalendarDate ,
			f.CalendarQuarter ,
			f.YearMonth ,
			f.SubsidiaryCode ,
			f.SubsidiaryName ,
			f.TPId ,
			f.TPNamee ,
			f.DeviceName,
			f.PFAMName ,
			f.ProductPartNbr ,
			f.SellThruQTY ,
			f.SellinQTY ,
			f.Year ,
			f.Month ,
			f.Day ,
			f.Day_of_Week ,
			f.COMM_IND ,
			f.Bundle_IND,
			f.Processor ,
			f.RAM ,
			f.QTR 
FROM FactTbl_may5_temp
WHERE dateid = 0




