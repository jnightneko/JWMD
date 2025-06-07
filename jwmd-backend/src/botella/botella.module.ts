import { Module } from '@nestjs/common';
import { BotellaController } from './botella.controller';
import { BotellaService } from './botella.service';
import { TypeOrmModule } from '@nestjs/typeorm';
import { Botella } from './botella.entity';

@Module({
  imports: [
    TypeOrmModule.forFeature([
      Botella
    ])
  ],
  controllers: [BotellaController],
  providers: [BotellaService]
})
export class BotellaModule {}
