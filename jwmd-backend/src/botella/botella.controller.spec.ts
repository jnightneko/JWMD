import { Test, TestingModule } from '@nestjs/testing';
import { BotellaController } from './botella.controller';

describe('BotellaController', () => {
  let controller: BotellaController;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [BotellaController],
    }).compile();

    controller = module.get<BotellaController>(BotellaController);
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });
});
