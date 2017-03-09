import sqlite3 as sql
import pandas.io.sql as pd_sql
import csv

def CombineSeedsOdds():
    con = sql.connect(":memory:")
    con.text_factory = str
    cur = con.cursor()
    cur.execute("CREATE TABLE seeds (Year, Team, Seed);")
    with open('Data\TournamentSeedsNumbers.csv', 'rb') as fin:
        dr = csv.DictReader(fin)
        to_db = [(i['Year'], i['Team'], i['Seed']) for i in dr]

    cur.executemany("INSERT INTO seeds (Year, Team, Seed) VALUES (?, ?, ?);", to_db)

    cur.execute("CREATE TABLE odds (Season, Team1, Team2, Odds1);")

    with open('Data\VegasOddsTeamNumbers.csv','rb') as fin:
        dr = csv.DictReader(fin)
        to_db = [(i['Season'],i['Team1'],i['Team2'],i['Odds1']) for i in dr]

    cur.executemany("INSERT INTO odds (Season, Team1, Team2, Odds1) VALUES (?, ?, ?, ?);", to_db)


    table = pd_sql.read_sql('SELECT DISTINCT o.Odds1, s1.Seed seed1, s2.Seed seed2 '
              'FROM seeds s1, seeds s2, odds o '
              'WHERE s1.Year = s2.Year AND s2.Year = o.Season AND o.Season = s1.Year '
              'AND s1.Team = o.Team1 AND s2.Team = o.Team2;', con)

    table.to_csv('Data\odds_seeds.csv')
    con.commit()
    con.close()



def CreateTable_WINSUM():
    con = sql.connect(":memory:")
    cur = con.cursor()
    cur.execute("CREATE TABLE winsum (Season,Daynum,Wteam,Wscore,Lteam,Lscore,ScoreDiff,Wloc,"
                "Numot,Wfgm,Wfga,Wfgm3,Wfga3,Wftm,Wfta,Wor,Wdr,Wast,Wto,Wstl,Wblk,Wpf,Lfgm,"
                "Lfga,Lfgm3,Lfga3,Lftm,Lfta,Lor,Ldr,Last,Lto,Lstl,Lblk,Lpf);")

    with open('Data\RegularSeasonDetailedResults.csv','rb') as fin:
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


def ReformatOriginal():
    con = sql.connect(":memory:")
    cur = con.cursor()
    cur.execute("CREATE TABLE averaged_stats (team, avg_score, avg_diff, avg_numot, avg_fgm, avg_fga, avg_fgm3, avg_ftm, "
                    "avg_fta, avg_or, avg_dr, avg_ast, avg_to, avg_stl, avg_blk, avg_pf);")

    with open('averaged_stats.csv', 'rb') as fin:
        dr = csv.DictReader(fin)
        to_db = [(i['team'], i['avg_score'], i['avg_diff'], i['avg_numot'], i['avg_fgm'], i['avg_fga'], i['avg_fgm3'],
                  i['avg_ftm'], i['avg_fta'], i['avg_or'], i['avg_dr'], i['avg_ast'], i['avg_to'],
                  i['avg_stl'], i['avg_blk'], i['avg_pf']) for i in dr]

    cur.executemany("INSERT INTO averaged_stats (team, avg_score, avg_diff, avg_numot, avg_fgm, avg_fga, avg_fgm3, avg_ftm, "
                    "avg_fta, avg_or, avg_dr, avg_ast, avg_to, avg_stl, avg_blk, avg_pf) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);", to_db)

    cur.execute("CREATE TABLE detailed_stats (Season,Daynum,Wteam,Wscore,Lteam,Lscore,ScoreDiff,Wloc,"
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

    cur.executemany("INSERT INTO detailed_stats (Season, Daynum, Wteam, Wscore, Lteam, Lscore, "
                    "ScoreDiff, Wloc, Numot, Wfgm, Wfga, Wfgm3, Wfga3, Wftm, Wfta, Wor, Wdr, Wast, "
                    "Wto, Wstl, Wblk, Wpf, Lfgm, Lfga, Lfgm3, Lfga3, Lftm, Lfta, Lor, Ldr, Last, Lto, "
                    "Lstl, Lblk, Lpf) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                    "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);", to_db)

    table = pd_sql.read_sql('SELECT DISTINCT dw.Wteam as team1, '
                            'dl.Lteam as team2, '
                            '1 as result, '
                            'dw.Daynum as daynum,'
                            'dw.Wloc as loc, '
                            'aw.avg_score as avg_score_1,'
                            'aw.avg_diff as avg_diff_1, '
                            'aw.avg_numot as avg_numot_1, '
                            'aw.avg_fgm as avg_fgm_1, '
                            'aw.avg_fga as avg_fga_1, '
                            'aw.avg_fgm3 as avg_fgm3_1, '
                            'aw.avg_ftm as avg_ftm_1, '
                            'aw.avg_fta as avg_fta_1, '
                            'aw.avg_or as avg_or_1, '
                            'aw.avg_dr as avg_dr_1, '
                            'aw.avg_ast as avg_ast_1, '
                            'aw.avg_to as avg_to_1, '
                            'aw.avg_stl as avg_stl_1, '
                            'aw.avg_blk as avg_blk_1, '
                            'aw.avg_pf as avg_pf_1, '
                            'al.avg_score as avg_score_2,'
                            'al.avg_diff as avg_diff_2, '
                            'al.avg_numot as avg_numot_2, '
                            'al.avg_fgm as avg_fgm_2, '
                            'al.avg_fga as avg_fga_2, '
                            'al.avg_fgm3 as avg_fgm3_2, '
                            'al.avg_ftm as avg_ftm_2, '
                            'al.avg_fta as avg_fta_2, '
                            'al.avg_or as avg_or_2, '
                            'al.avg_dr as avg_dr_2, '
                            'al.avg_ast as avg_ast_2, '
                            'al.avg_to as avg_to_2, '
                            'al.avg_stl as avg_stl_2, '
                            'al.avg_blk as avg_blk_2, '
                            'al.avg_pf as avg_pf_2 '
                            'FROM  detailed_stats dw, detailed_stats dl, averaged_stats aw, averaged_stats al '
                            'WHERE dw.Wteam = aw.team '
                            'AND dl.Lteam = al.team '
                            'AND (dw.Season = \'2016\') '
                            'AND (dl.Season = \'2016\') '
                            'AND dw.Wteam = dl.Wteam '
                            'AND dw.Lteam = dl.Lteam; ', con)

    table.to_csv('RegularizedSeasonDetailedWtoL.csv')

    table = pd_sql.read_sql('SELECT DISTINCT dw.Wteam as team1, '
                            'dl.Lteam as team2, '
                            '0 as result, '
                            'dw.Daynum as daynum,'
                            '-1 * dw.Wloc as loc, '
                            'al.avg_score as avg_score_1,'
                            'al.avg_diff as avg_diff_1, '
                            'al.avg_numot as avg_numot_1, '
                            'al.avg_fgm as avg_fgm_1, '
                            'al.avg_fga as avg_fga_1, '
                            'al.avg_fgm3 as avg_fgm3_1, '
                            'al.avg_ftm as avg_ftm_1, '
                            'al.avg_fta as avg_fta_1, '
                            'al.avg_or as avg_or_1, '
                            'al.avg_dr as avg_dr_1, '
                            'al.avg_ast as avg_ast_1, '
                            'al.avg_to as avg_to_1, '
                            'al.avg_stl as avg_stl_1, '
                            'al.avg_blk as avg_blk_1, '
                            'al.avg_pf as avg_pf_1, '
                            'aw.avg_score as avg_score_2,'
                            'aw.avg_diff as avg_diff_2, '
                            'aw.avg_numot as avg_numot_2, '
                            'aw.avg_fgm as avg_fgm_2, '
                            'aw.avg_fga as avg_fga_2, '
                            'aw.avg_fgm3 as avg_fgm3_2, '
                            'aw.avg_ftm as avg_ftm_2, '
                            'aw.avg_fta as avg_fta_2, '
                            'aw.avg_or as avg_or_2, '
                            'aw.avg_dr as avg_dr_2, '
                            'aw.avg_ast as avg_ast_2, '
                            'aw.avg_to as avg_to_2, '
                            'aw.avg_stl as avg_stl_2, '
                            'aw.avg_blk as avg_blk_2, '
                            'aw.avg_pf as avg_pf_2 '
                            'FROM  detailed_stats dw, detailed_stats dl, averaged_stats aw, averaged_stats al '
                            'WHERE dw.Wteam = aw.team '
                            'AND dl.Lteam = al.team '
                            'AND (dw.Season = \'2016\') '
                            'AND (dl.Season = \'2016\') '
                            'AND dw.Wteam = dl.Wteam '
                            'AND dw.Lteam = dl.Lteam; ', con)
    table.to_csv('RegularizedSeasonDetailedLtoW.csv')
    con.commit()
    con.close()

def ReformatTournamentData():


def main():
    step = "Learn Vegas Odds"
    if step == "Regularize Regular Season" :
        CreateTable_WINSUM()
        CreateTable_LOSESUM()
        CombineTables()
        ReformatOriginal()
    elif step == "Regularize Tournament Data" :
        ReformatTournamentData()
    elif step == "Seed-Tournament Table":
        return
    elif step == "Learn Vegas Odds":
        CombineSeedsOdds()




if __name__ == "__main__":
    main()



