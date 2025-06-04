import { Body, Controller, Delete, Get, Param, Post, Put } from '@nestjs/common';
import { BotellaService } from './botella.service';
import { Botella } from './botella.entity';

@Controller('botella')
export class BotellaController {
    constructor(
        private readonly clienteService: BotellaService
    ) {}

    @Get()
    findAll() {
        return this.clienteService.findAll()
    }

    @Get(':id')
    findOne(@Param('id') id: string) {
        return this.clienteService.findOne(id)
    }

    @Post()
    create(@Body() objetoTarjeta: Partial<Botella>) {
        return this.clienteService.create(objetoTarjeta)
    }

    @Put(':id')
    update(@Param('id') id: string, @Body() objetoTarjeta: Partial<Botella>) {
        return this.clienteService.update(id, objetoTarjeta)
    }

    @Delete(':id')
    delete(@Param('id') id: string) {
        return this.clienteService.delete(id)
    }
}

