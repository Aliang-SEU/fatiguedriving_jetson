#ifndef DRIVERDATABASE_H
#define DRIVERDATABASE_H

#include <QSqlDatabase>
#include <QSqlError>
#include <QSqlQuery>
#include <QDateTime>

class DriverDatabase
{
public:
    DriverDatabase();
    ~DriverDatabase();
    bool addNewRecord(QString filename, QString username="annoymous");
    void queryAllRecord();
    void deleteRecord(int id);

private:
    QSqlDatabase dataBase;
};

#endif // DRIVERDATABASE_H
