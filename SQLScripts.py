import sqlite3 as sql
import pandas.io.sql as pd_sql
import csv

def CreateTable_WINSUM():
    con = sql.connect(":memory:")
    cur = con.cursor()
    cur.execute("CREATE TABLE winsum (Season,Daynum,Wteam,Wscore,Lteam,Lscore,ScoreDiff,Wloc,"
                "Numot,Wfgm,Wfga,Wfgm3,Wfga3,Wftm,Wfta,Wor,Wdr,Wast,Wto,Wstl,Wblk,Wpf,Lfgm,"
                "Lfga,Lfgm3,Lfga3,Lftm,Lfta,Lor,Ldr,Last,Lto,Lstl,Lblk,Lpf);")

    with open('RegularSeasonDetailedResults.csv','rb') as fin:
        dr = csv.DictReader(fin)
        to_db = [(i['Season'],i['Daynum'],i['Wteam'],i['Wscore'],i['Lteam'],
                  i['Lscore'],i['ScoreDiff'],i['Wloc'],i['Numot'],i['Wfgm'],
                  i['Wfga'],i['Wfgm3'],i['Wfga3'],i['Wftm'],i['Wfta'],i['Wor'],i['Wdr'],
                  i['Wast'],i['Wto'],i['Wstl'],i['Wblk'],i['Wpf'],i['Lfgm'],i['Lfga'],
                  i['Lfgm3'],i['Lfga3'],i['Lftm'],i['Lfta'],i['Lor'],i['Ldr'],i['Last'],
                  i['Lto'],i['Lstl'],i['Lblk'],i['Lpf']) for i in dr]

    cur.executemany("INSERT INTO winsum (Season, Daynum, Wteam, Wscore, Lteam, Lscore, "
                    "ScoreDiff, Wloc, Numot, Wfgm, Wfga, Wfgm3, Wfga3, Wftm, Wfta, Wor, Wdr, Wast, "
                    "Wto, Wstl, Wblk, Wpf, Lfgm, Lfga, Lfgm3, Lfga3, Lftm, Lfta, Lor, Ldr, Last, Lto, "
                    "Lstl, Lblk, Lpf) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                    "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);", to_db)


    table = pd_sql.read_sql('SELECT Wteam as team, COUNT(Wteam) as count, SUM(Wscore) as score,'
              'SUM(ScoreDiff) as diff, SUM(Numot) as Numot, '
              'SUM(Wfgm) as fgm, SUM(Wfga) as fga, SUM(Wfgm3) as fgm3,'
              'SUM(Wftm) as ftm, SUM(Wfta) as fta, SUM(Wor) as or2,'
              'SUM(Wdr) as dr, SUM(Wast) as ast, SUM(Wto) as to2, '
              'SUM(Wstl) as stl, SUM(Wblk) as blk, SUM(Wpf) as pf '
              'FROM winsum '
              'WHERE (Season = \'2016\') '
              'GROUP BY Wteam;', con)

    table.to_csv('win_sum.csv')
    con.commit()
    con.close()


def CreateTable_LOSESUM():
    con = sql.connect(":memory:")
    cur = con.cursor()
    cur.execute("CREATE TABLE losesum (Season,Daynum,Wteam,Wscore,Lteam,Lscore,ScoreDiff,Wloc,"
                "Numot,Wfgm,Wfga,Wfgm3,Wfga3,Wftm,Wfta,Wor,Wdr,Wast,Wto,Wstl,Wblk,Wpf,Lfgm,"
                "Lfga,Lfgm3,Lfga3,Lftm,Lfta,Lor,Ldr,Last,Lto,Lstl,Lblk,Lpf);")

    with open('RegularSeasonDetailedResults.csv', 'rb') as fin:
        dr = csv.DictReader(fin)
        to_db = [(i['Season'], i['Daynum'], i['Wteam'], i['Wscore'], i['Lteam'],
                  i['Lscore'], i['ScoreDiff'], i['Wloc'], i['Numot'], i['Wfgm'],
                  i['Wfga'], i['Wfgm3'], i['Wfga3'], i['Wftm'], i['Wfta'], i['Wor'], i['Wdr'],
                  i['Wast'], i['Wto'], i['Wstl'], i['Wblk'], i['Wpf'], i['Lfgm'], i['Lfga'],
                  i['Lfgm3'], i['Lfga3'], i['Lftm'], i['Lfta'], i['Lor'], i['Ldr'], i['Last'],
                  i['Lto'], i['Lstl'], i['Lblk'], i['Lpf']) for i in dr]

    cur.executemany("INSERT INTO losesum (Season, Daynum, Wteam, Wscore, Lteam, Lscore, "
                    "ScoreDiff, Wloc, Numot, Wfgm, Wfga, Wfgm3, Wfga3, Wftm, Wfta, Wor, Wdr, Wast, "
                    "Wto, Wstl, Wblk, Wpf, Lfgm, Lfga, Lfgm3, Lfga3, Lftm, Lfta, Lor, Ldr, Last, Lto, "
                    "Lstl, Lblk, Lpf) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                    "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);", to_db)

    table = pd_sql.read_sql('SELECT Lteam as team, COUNT(Lteam) as count, SUM(Lteam) as score,'
                            '-1 * SUM(ScoreDiff) as diff, SUM(Numot) as Numot, '
                            'SUM(Lfgm) as fgm, SUM(Lfga) as fga, SUM(Lfgm3) as fgm3,'
                            'SUM(Lftm) as ftm, SUM(Lfta) as fta, SUM(Lor) as or2,'
                            'SUM(Ldr) as dr, SUM(Last) as ast, SUM(Lto) as to2, '
                            'SUM(Lstl) as stl, SUM(Lblk) as blk, SUM(Lpf) as pf '
                            'FROM losesum '
                            'WHERE (Season = \'2016\') '
                            'GROUP BY Lteam;', con)

    table.to_csv('lose_sum.csv')
    con.commit()
    con.close()

def CombineTables():
        con = sql.connect(":memory:")
        cur = con.cursor()
        cur.execute("CREATE TABLE stats (team, count, score, diff, Numot, fgm, fga, fgm3, "
                    "ftm, fta, or2, dr, ast, to2, stl, blk, pf);")

        with open('win_sum.csv', 'rb') as fin:
            dr = csv.DictReader(fin)
            to_db = [(i['team'], i['count'], i['score'], i['diff'], i['Numot'], i['fgm'], i['fga'],
                      i['fgm3'], i['ftm'], i['fta'], i['or2'], i['dr'], i['ast'],
                      i['to2'], i['stl'], i['blk'], i['pf']) for i in dr]

        cur.executemany("INSERT INTO stats (team, count, score, diff, Numot, fgm, fga, fgm3, "
                    "ftm, fta, or2, dr, ast, to2, stl, blk, pf) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);", to_db)

        with open('lose_sum.csv', 'rb') as fin:
            dr = csv.DictReader(fin)
            to_db = [(i['team'], i['count'], i['score'], i['diff'], i['Numot'], i['fgm'], i['fga'],
                      i['fgm3'], i['ftm'], i['fta'], i['or2'], i['dr'], i['ast'],
                      i['to2'], i['stl'], i['blk'], i['pf']) for i in dr]

        cur.executemany("INSERT INTO stats (team, count, score, diff, Numot, fgm, fga, fgm3, "
                    "ftm, fta, or2, dr, ast, to2, stl, blk, pf) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);", to_db)


        table = pd_sql.read_sql('SELECT team, '
                            'CAST(SUM(score) AS float) / CAST(SUM(count) AS float) as avg_score, '
                            'CAST(SUM(diff) AS float) / CAST(SUM(count) AS float) as avg_diff, '
                            'CAST(SUM(Numot) AS float) / CAST(SUM(count) AS float) as avg_numot, '
                            'CAST(SUM(fgm) AS float) / CAST(SUM(count) AS float) as avg_fgm, '
                            'CAST(SUM(fga) AS float) / CAST(SUM(count) AS float) as avg_fga, '
                            'CAST(SUM(fgm3) AS float) / CAST(SUM(count) AS float) as avg_fgm3, '
                            'CAST(SUM(ftm) AS float) / CAST(SUM(count) AS float) as avg_ftm, '
                            'CAST(SUM(fta) AS float) / CAST(SUM(count) AS float) as avg_fta, '
                            'CAST(SUM(or2) AS float) / CAST(SUM(count) AS float) as avg_or,'
                            'CAST(SUM(dr) AS float) / CAST(SUM(count) AS float) as avg_dr, '
                            'CAST(SUM(ast) AS float) / CAST(SUM(count) AS float) as avg_ast, '
                            'CAST(SUM(to2) AS float) / CAST(SUM(count) AS float) as avg_to, '
                            'CAST(SUM(stl) AS float) / CAST(SUM(count) AS float) as avg_stl, '
                            'CAST(SUM(blk) AS float) / CAST(SUM(count) AS float) as avg_blk, '
                            'CAST(SUM(pf) AS float) / CAST(SUM(count) AS float) as avg_pf '
                            'FROM stats '
                            'GROUP BY team;', con)

        table.to_csv('averaged_stats.csv')
        con.commit()
        con.close()


"""CreateTable_WINSUM()"""
"""CreateTable_LOSESUM()"""
CombineTables()

