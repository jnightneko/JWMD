import { Test, TestingModule } from '@nestjs/testing';
import { BotellaService } from './botella.service';

describe('BotellaService', () => {
  let service: BotellaService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [BotellaService],
    }).compile();

    service = module.get<BotellaService>(BotellaService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });
});
