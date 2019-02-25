#include "DriverDatabase.h"
#include <QDebug>
DriverDatabase::DriverDatabase(){

    dataBase = QSqlDatabase::addDatabase("QMYSQL"); //使用mysql数据库
    dataBase.setHostName("127.0.0.1");  //本地数据库
    dataBase.setUserName("root");   //数据库的帐号
    dataBase.setPassword("1234"); //数据库的密码
    dataBase.setDatabaseName("fatiguedriving");    //数据库名称
    dataBase.open();
    if(!dataBase.isOpen()) {
        qDebug() << "Error:数据库连接错误! 原因:" << dataBase.lastError().text();
    }else {
        qDebug() << "数据库连接成功！";
    }

}

DriverDatabase::~DriverDatabase(){
    dataBase.close();
}

/** 插入一条记录
 * @brief DriverDatabase::addNewRecord
 * @return
 */
bool DriverDatabase::addNewRecord(QString filename, QString username){
    //首先检查表是否存在
    QSqlQuery sqlQuery(dataBase);
//    sqlQuery.exec("SET NAMES 'Latin1'");   //支持中文
//    sqlQuery.exec("CREATE TABLE driving_record(id int primary key auto_increment,"
//               "username varchar(20),"
//               "createtime timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,"
//               "filename varchar(200)))");
    QString insertSql = QString("INSERT INTO driving_record(username, filename)");
    insertSql += QString("VALUES('%1','%2')").arg(username).arg(filename);
    bool ok = sqlQuery.exec(insertSql);
    if(ok) {
        qDebug() << "insert success!";
    }else {
        qDebug() << "insert failed!";
    }
}

/** 查询所有的疲劳驾驶记录
 * @brief DriverDatabase::queryAllRecord
 */
void DriverDatabase::queryAllRecord() {
    QSqlQuery sqlQuery(dataBase);
    sqlQuery.exec("select * from driving_record;");
    while(sqlQuery.next()) {
        QDateTime time = sqlQuery.value(2).toDateTime();
        qDebug() << sqlQuery.value(0).toInt() << " | " << sqlQuery.value(1).toString() << " | " << time.toString("yyyy/mm/dd hh:mm:ss") << " | " << sqlQuery.value(3).toString() << endl;

    }
}


/** 删除某个疲劳记录
 * @brief DriverDatabase::deleteRecord
 */
void DriverDatabase::deleteRecord(int id) {
    QSqlQuery sqlQuery(dataBase);
    sqlQuery.exec(QString("delect from driving_record where id = %1;").arg(id));
    qDebug() << sqlQuery.lastError() << endl;
}
