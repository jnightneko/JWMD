import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { BotellaModule } from './botella/botella.module';
import { Botella } from './botella/botella.entity';
import { TypeOrmModule } from '@nestjs/typeorm';

@Module({
  imports: [
    TypeOrmModule.forRoot({
      type: 'mysql',
      host: 'localhost',
      port: 3306,
      username: 'root',
      password: '1234',
      database: 'jwmd',
      entities: [Botella], //Poner todas las entidades que nos van a servir
      synchronize: true
    }),
  BotellaModule],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
